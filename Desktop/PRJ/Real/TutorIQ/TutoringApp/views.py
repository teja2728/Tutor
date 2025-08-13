from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
from django.http import HttpResponse
from django.conf import settings
import os
import pymysql
from django.core.files.storage import FileSystemStorage
from datetime import date
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from numpy import dot
from numpy.linalg import norm
from string import punctuation
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import random

import os
from tensorflow.keras.utils import to_categorical
from keras.layers import MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D, BatchNormalization
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
import pickle

from sklearn.model_selection import train_test_split
from sklearn import svm #SVM class
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from django.views.decorators.csrf import csrf_exempt
import speech_recognition as sr
import subprocess

# Define database connection parameters from settings
DB_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'user': 'tutoriq_user',
    'password': 'secure_password',
    'database': 'tutoriq_db',
    'charset': 'utf8'
}

# Initialize global variables
global username, X, Y, vectorizer, tfidf, material, labels, label_encoder, rf_cls
username = None
X = None
Y = None
vectorizer = None
tfidf = None
material = None
labels = None
label_encoder = None
rf_cls = None

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()
accuracy = []
precision = []
recall = []
fscore = []
recognizer = sr.Recognizer()

def cleanText(doc):
    tokens = doc.split()
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    tokens = [ps.stem(token) for token in tokens]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = ' '.join(tokens)
    return tokens

def calculateMetrics(algorithm, testY, predict):
    p = precision_score(testY, predict,average='macro') * 100
    r = recall_score(testY, predict,average='macro') * 100
    f = f1_score(testY, predict,average='macro') * 100
    a = accuracy_score(testY,predict)*100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)

def SearchTutorial(request):
    if request.method == 'GET':
        return render(request, 'SearchTutorial.html', {})

def getMaterialName(material_type):
    value = []
    con = pymysql.connect(**DB_CONFIG)
    with con:
        cur = con.cursor()
        cur.execute("select material_desc,material_name FROM uploadmaterial where material_type=%s", (material_type,))
        rows = cur.fetchall()
        for row in rows:
            value.append([row[0],row[1]])
    return value       

def DownloadDataAction(request):
    if request.method == 'GET':
        global username
        filename = request.GET['t1']
        with open("TutoringApp/static/material/"+filename, "rb") as myfile:
            data = myfile.read()
        myfile.close()
        filename = filename
        #file_mimetype = mimetypes.guess_type(filename)
        response = HttpResponse(data,content_type="application/octet-stream")
        #response['X-Sendfile'] = filename
        #response['Content-Length'] = os.stat("DecentralizedApp/static/"+filename).st_size
        response['Content-Disposition'] = "attachment; filename=%s" % filename
        return response

def initialize_model():
    """Initialize the ML model and vectorizer"""
    global X, Y, vectorizer, tfidf, material, labels, label_encoder, rf_cls
    try:
        label_encoder = LabelEncoder()
        data = []
        material = []
        
        # Connect to database and get training data
        con = pymysql.connect(**DB_CONFIG)
        with con:
            cur = con.cursor()
            cur.execute("select * FROM uploadmaterial")
            rows = cur.fetchall()
            for row in rows:
                data.append([row[1].strip().lower(), row[3].strip().lower()])
                material.append(row[2].strip())
        
        if not data:  # If no data in database
            print("No training data available")
            return False
            
        data = pd.DataFrame(data, columns=['label', 'material'])
        labels = np.unique(data['label'])
        data['label'] = pd.Series(label_encoder.fit_transform(data['label'].astype(str)))
        
        mat = []
        data = data.values
        X = []
        Y = []
        
        # Process training data
        for i in range(len(data)):
            lbl = data[i,0]
            content = data[i,1]
            content = cleanText(content)
            X.append(content)
            Y.append(lbl)
            name = material[i]
            mat.append(name)
            
            # Data augmentation
            for j in range(0, 3):
                arr = content.split(" ")
                size = random.randint(1, len(arr))
                temp = " ".join(arr[:size])
                X.append(temp)
                Y.append(lbl)
                mat.append(name)
        
        material = mat
        X = np.asarray(X)
        Y = np.asarray(Y)
        material = np.asarray(material)
        
        # Initialize and fit vectorizer
        vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=False, norm=None, decode_error='replace')
        tfidf = vectorizer.fit_transform(X).toarray()
        
        # Shuffle data
        indices = np.arange(tfidf.shape[0])
        np.random.shuffle(indices)
        tfidf = tfidf[indices]
        Y = Y[indices]
        material = material[indices]
        
        # Train Random Forest model
        X_train, X_test, y_train, y_test = train_test_split(tfidf, Y, test_size=0.2)
        rf_cls = RandomForestClassifier(max_depth=10)
        rf_cls.fit(X_train, y_train)
        
        print("Model initialized successfully")
        return True
        
    except Exception as e:
        print(f"Error initializing model: {str(e)}")
        return False

def SearchTutorialAction(request):
    if request.method == 'POST':
        global X, Y, vectorizer, tfidf, material, labels, label_encoder, rf_cls
        
        # Initialize model if not already done
        if vectorizer is None:
            if not initialize_model():
                return render(request, 'UserScreen.html', {'data': 'Error: Model not initialized. Please contact administrator.'})
        
        query = request.POST.get('t1', False)
        if not query:
            return render(request, 'UserScreen.html', {'data': 'Please enter a search query'})
            
        query = query.strip().lower()
        query = cleanText(query)
        query = vectorizer.transform([query]).toarray()
        testData = query[0]
        
        # Find best matching tutorial
        max_accuracy = 0
        solution_index = -1
        for i in range(len(tfidf)):
            predict_score = dot(tfidf[i], testData)/(norm(tfidf[i])*norm(testData))
            if predict_score > max_accuracy:
                max_accuracy = predict_score
                solution_index = i
        
        if solution_index != -1 and max_accuracy > 0.3:  # Threshold for matching
            predict = Y[solution_index]
            pred = labels[predict]
            predict = getMaterialName(pred)
            
            output = '<table border=1 align=center>'
            output+='<tr><th><font size=3 color=black>Predicted Tutoring Type</font></th>'
            output+='<th><font size=3 color=black>Description</font></th>'
            output+='<th><font size=3 color=black>Tutorial Name</font></th>'
            output+='<th><font size=3 color=black>Download Tutorial</font></th></tr>'
            
            for i in range(len(predict)):
                desc = predict[i][0]
                name = predict[i][1]
                output+='<tr><td><font color="black" size="3">'+pred+'</td>'
                output+='<td><font color="black" size="3">'+desc+'</td>'
                output+='<td><font color="black" size="3">'+name+'</td>'
                output+='<td><a href=\'DownloadDataAction?t1='+name+'\'><font size=3 color=black>Click Here</font></a></td></tr>'
            
            output+='</table>'
            context = {'data': output}
        else:
            context = {'data': 'No matching tutorials found. Please try a different search term.'}
            
        return render(request, 'UserScreen.html', context)    
         
def SearchTutorialVoice(request):
    if request.method == 'GET':
        return render(request, 'SearchTutorialVoice.html', {})

@csrf_exempt
def SearchTutorialVoiceAction(request):
    if request.method == "POST":
        global X, Y, vectorizer, tfidf, material, labels, label_encoder, rf_cls
        
        # Initialize model if not already done
        if vectorizer is None:
            if not initialize_model():
                return HttpResponse("Error: Model not initialized. Please contact administrator.", status=500)
        
        try:
            audio_data = request.FILES.get('data')
            if not audio_data:
                return HttpResponse("No audio data received", status=400)

            fs = FileSystemStorage()
            
            # Clean up old files
            for old_file in ['record.wav', 'record1.wav']:
                old_path = os.path.join('TutoringApp/static', old_file)
                if os.path.exists(old_path):
                    os.remove(old_path)
            
            # Save the new audio file
            fs.save('TutoringApp/static/record.wav', audio_data)
            
            # Convert audio format using ffmpeg
            try:
                ffmpeg_path = os.path.join('TutoringApp/static', 'ffmpeg.exe')
                input_path = os.path.join('TutoringApp/static', 'record.wav')
                output_path = os.path.join('TutoringApp/static', 'record1.wav')
                
                subprocess.run([
                    ffmpeg_path,
                    '-i', input_path,
                    '-acodec', 'pcm_s16le',
                    '-ar', '16000',
                    '-ac', '1',
                    output_path
                ], check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                print(f"FFmpeg error: {e.stderr.decode()}")
                return HttpResponse("Error processing audio", status=500)

            # Perform speech recognition
            with sr.AudioFile(output_path) as source:
                audio = recognizer.record(source)
                try:
                    text = recognizer.recognize_google(audio)
                except sr.UnknownValueError:
                    return HttpResponse("Could not understand audio. Please speak clearly and try again.")
                except sr.RequestError as e:
                    print(f"Speech recognition error: {e}")
                    return HttpResponse("Error accessing speech recognition service. Please try again later.")

            if not text:
                return HttpResponse("No speech detected. Please try again.")

            # Process the recognized text
            query = text.strip().lower()
            query = cleanText(query)
            query = vectorizer.transform([query]).toarray()
            testData = query[0]

            # Find best matching tutorial
            max_accuracy = 0
            solution_index = -1
            for i in range(len(tfidf)):
                predict_score = dot(tfidf[i], testData)/(norm(tfidf[i])*norm(testData))
                if predict_score > max_accuracy:
                    max_accuracy = predict_score
                    solution_index = i

            if solution_index != -1 and max_accuracy > 0.3:  # Threshold for matching
                predict = Y[solution_index]
                pred = labels[predict]
                predict = getMaterialName(pred)
                
                if predict:
                    desc = predict[0][0]
                    name = predict[0][1]
                    return HttpResponse(f"{desc}#http://127.0.0.1:8000/DownloadDataAction?t1={name}")
                else:
                    return HttpResponse("No matching tutorial found. Please try a different search term.")
            else:
                return HttpResponse("No matching tutorial found. Please try a different search term.")

        except Exception as e:
            print(f"Error in voice search: {str(e)}")
            return HttpResponse("An error occurred while processing your request. Please try again.", status=500)


def TrainML(request):
    if request.method == 'GET':
        global X, Y, vectorizer, tfidf, material, labels, label_encoder, rf_cls
        global accuracy, precision, recall, fscore
        accuracy.clear()
        precision.clear()
        recall.clear()
        fscore.clear()                                          
        label_encoder = LabelEncoder()
        data = []
        material = []
        con = pymysql.connect(**DB_CONFIG)
        with con:
            cur = con.cursor()
            cur.execute("select * FROM uploadmaterial")
            rows = cur.fetchall()
            for row in rows:
                data.append([row[1].strip().lower(), row[3].strip().lower()])
                material.append(row[2].strip())
        data = pd.DataFrame(data, columns=['label', 'material'])
        labels = np.unique(data['label'])
        data['label'] = pd.Series(label_encoder.fit_transform(data['label'].astype(str)))#encode all str columns to numeric
        mat = []
        data = data.values
        X = []
        Y = []
        for i in range(len(data)):
            lbl = data[i,0]
            content = data[i,1]
            content = cleanText(content)
            print(content)
            X.append(content)
            Y.append(lbl)
            name = material[i]
            mat.append(name)
            for j in range(0, 3):
                arr = content.split(" ")
                size = random.randint(1, len(arr))
                temp = ""
                for k in range(0, size):
                    temp += arr[k]+" "
                temp = temp.strip()
                X.append(temp)
                Y.append(lbl)
                mat.append(name)
        material = mat
        X = np.asarray(X)
        Y = np.asarray(Y)
        material = np.asarray(material)        
    vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=False, norm=None, decode_error='replace')
    tfidf = vectorizer.fit_transform(X).toarray()
    indices = np.arange(tfidf.shape[0])
    np.random.shuffle(indices)#shuffling dataset values
    tfidf = tfidf[indices]
    Y = Y[indices]
    material = material[indices]
    print(Y)
    print(tfidf)
    X_train, X_test, y_train, y_test = train_test_split(tfidf, Y, test_size = 0.2)

    #training and evaluating performance of SVM algorithm
    svm_cls = svm.SVC()
    svm_cls.fit(X_train, y_train)#train algorithm using training features and target value
    predict = svm_cls.predict(X_test) #perform prediction on test data
    #call this function with true and predicted values to calculate accuracy and other metrics
    calculateMetrics("SVM Algorithm", y_test, predict)

    #training and evaluating performance of decision tree algorithm
    dt_cls = DecisionTreeClassifier()
    dt_cls.fit(X_train, y_train)#train algorithm using training features and target value
    predict =dt_cls.predict(X_test)#perform prediction on test data
    #call this function with true and predicted values to calculate accuracy and other metrics
    calculateMetrics("Decision Tree Algorithm", y_test, predict)

    rf_cls = RandomForestClassifier(max_depth=10)
    rf_cls.fit(X_train, y_train)#train algorithm using training features and target value
    predict = rf_cls.predict(X_test)#perform prediction on test data
    #call this function with true and predicted values to calculate accuracy and other metrics
    calculateMetrics("Random Forest", y_test, predict)

    nb_cls = GaussianNB()
    nb_cls.fit(X_train, y_train)#train algorithm using training features and target value
    predict = nb_cls.predict(X_test)#perform prediction on test data
    #call this function with true and predicted values to calculate accuracy and other metrics
    calculateMetrics("Naive Bayes", y_test, predict)

    X_train1 = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1, 1))
    X_test1 = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1, 1))
    y_train1 = to_categorical(y_train)
    y_test1 = to_categorical(y_test)
    #creating deep learning cnn model object
    '''cnn_model = Sequential()
    #defining CNN layer wwith 32 neurons of size 1 X 1 to filter dataset features 32 times
    cnn_model.add(Convolution2D(32, (1 , 1), input_shape = (X_train1.shape[1], X_train1.shape[2], X_train1.shape[3]), activation = 'relu'))
    #defining maxpool layet to collect relevant filtered features from previous CNN layer
    cnn_model.add(MaxPooling2D(pool_size = (1, 1)))
    #creating another CNN layer with 16 neurons to optimzed features 16 times
    cnn_model.add(Convolution2D(16, (1, 1), activation = 'relu'))
    #max layet to collect relevant features
    cnn_model.add(MaxPooling2D(pool_size = (1, 1)))
    #convert multidimension features to single flatten size
    cnn_model.add(Flatten())
    #define output prediction layer
    cnn_model.add(Dense(units = 256, activation = 'relu'))
    cnn_model.add(Dense(units = y_train1.shape[1], activation = 'softmax'))
    #compile, train and load CNN model
    cnn_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    if os.path.exists("model/cnn_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/cnn_weights.hdf5', verbose = 1, save_best_only = True)
        hist = cnn_model.fit(X_train1, y_train1, batch_size = 8, epochs = 10, validation_data=(X_test1, y_test1), callbacks=[model_check_point], verbose=1)
        f = open('model/cnn_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        cnn_model.load_weights("model/cnn_weights.hdf5")
    #perform prediction on test data   
    predict = cnn_model.predict(X_test1)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(y_test1, axis=1)
    #call this function to calculate accuracy and other metrics
    calculateMetrics("CNN", y_test1, predict)'''

    algorithms = ['SVM', 'Decision Tree', 'Random Forest', 'Naive Bayes']
    output = '<table border=1 align=center>'
    output+='<tr><th><font size=3 color=black>Algorithm Names</font></th>'
    output+='<th><font size=3 color=black>Accuracy</font></th>'
    output+='<th><font size=3 color=black>Precision</font></th>'
    output+='<th><font size=3 color=black>Recall</font></th>'
    output+='<th><font size=3 color=black>FSCORE</font></th></tr>'
    for i in range(len(algorithms)):
        output+='<tr><td><font color="black" size="3">'+algorithms[i]+'</td>'
        output+='<td><font color="black" size="3">'+str(accuracy[i])+'</td>'
        output+='<td><font color="black" size="3">'+str(precision[i])+'</td>'
        output+='<td><font color="black" size="3">'+str(recall[i])+'</td>'
        output+='<td><font color="black" size="3">'+str(fscore[i])+'</td></tr>'
    context= {'data':output}
    return render(request, 'AdminScreen.html', context)    


def UploadMaterial(request):
    if request.method == 'GET':
        return render(request, 'UploadMaterial.html', {})

def AdminScreen(request):
    return render(request, 'AdminScreen.html')


def UploadMaterialAction(request):
    if request.method == 'POST':
        global username
        material_type = request.POST.get('t1', False)
        desc = request.POST.get('t2', False)
        data = request.FILES['t3'].read()
        name = request.FILES['t3'].name
        today = str(date.today())
        with open('TutoringApp/static/material/'+name, "wb") as file:
            file.write(data)
        file.close()
        material_id = 0
        con = pymysql.connect(**DB_CONFIG)
        with con:
            cur = con.cursor()
            cur.execute("select max(material_id) FROM uploadmaterial")
            rows = cur.fetchall()
            for row in rows:
                material_id = row[0]
        if material_id is None:
            material_id = 1
        else:
            material_id = material_id + 1
        db_connection = pymysql.connect(**DB_CONFIG)
        db_cursor = db_connection.cursor()
        student_sql_query = "INSERT INTO uploadmaterial(material_id, material_type, material_name, material_desc, upload_date) VALUES(%s, %s, %s, %s, %s)"
        db_cursor.execute(student_sql_query, (str(material_id), material_type, name, desc, today))
        db_connection.commit()
        print(db_cursor.rowcount, "Record Inserted")
        if db_cursor.rowcount == 1:
            context= {'data':'Tutoring Material Saved at server with ID : '+str(material_id)}
            return render(request, 'UploadMaterial.html', context)
        else:
            context= {'data':'Error in saving tutoring material'}
            return render(request, 'UploadMaterial.html', context)
        

def UserLoginAction(request):
    global username
    if request.method == 'POST':
        user = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        status = "failed"
        con = pymysql.connect(**DB_CONFIG)
        with con:
            cur = con.cursor()
            cur.execute("select username,password FROM register")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == user and row[1] == password:
                    status = 'success'
                    break
        if status == 'success':
            username = user
            status = 'Welcome : '+username
            context= {'data':status}
            return render(request, 'UserScreen.html', context)
        else:
            context= {'data':'invalid login'}
            return render(request, 'UserLogin.html', context)

def AdminLoginAction(request):
    global username
    if request.method == 'POST':
        user = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        if user == 'admin' and password == 'admin':
            username = user
            status = 'Welcome : '+username
            context= {'data':status}
            return render(request, 'AdminScreen.html', context)
        else:
            context= {'data':'invalid login'}
            return render(request, 'AdminLogin.html', context)        

def Register(request):
    if request.method == 'GET':
       return render(request, 'Register.html', {})

def index(request):
    if request.method == 'GET':
        return render(request, 'index.html', {})    

def logout(request):
    if request.method == 'GET':
        return render(request, 'index.html', {})

def AdminLogin(request):
    if request.method == 'GET':
       return render(request, 'AdminLogin.html', {})

def UserLogin(request):
    if request.method == 'GET':
       return render(request, 'UserLogin.html', {})    

def RegisterAction(request):
    if request.method == 'POST':
        global username
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        gender = request.POST.get('gender', False)
        qualification = request.POST.get('qualification', False)
        contact = request.POST.get('t3', False)
        email = request.POST.get('t4', False)
        address = request.POST.get('t5', False)
                
        output = "none"
        con = pymysql.connect(**DB_CONFIG)
        with con:
            cur = con.cursor()
            cur.execute("select username FROM register")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == username:
                    output = username+" Username already exists"                    
        if output == "none":                      
            db_connection = pymysql.connect(**DB_CONFIG)
            db_cursor = db_connection.cursor()
            student_sql_query = "INSERT INTO register(username,password,gender,qualification,contact,email,address) VALUES(%s, %s, %s, %s, %s, %s, %s)"
            db_cursor.execute(student_sql_query, (username, password, gender, qualification, contact, email, address))
            db_connection.commit()
            print(db_cursor.rowcount, "Record Inserted")
            if db_cursor.rowcount == 1:
                context= {'data':'Signup Process Completed'}
                return render(request, 'Register.html', context)
            else:
                context= {'data':'Error in signup process'}
                return render(request, 'Register.html', context)
        else:
            context= {'data':output}
            return render(request, 'Register.html', context)