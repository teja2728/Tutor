from django.urls import path

from . import views

urlpatterns = [path("", views.index, name="index"),
	       path("AdminLoginAction", views.AdminLoginAction, name="AdminLoginAction"),
	       path("AdminLogin", views.AdminLogin, name="AdminLogin"),
           path("AdminScreen", views.AdminScreen, name="AdminScreen"),
	       path("UserLogin", views.UserLogin, name="UserLogin"),
	       path("UserLoginAction", views.UserLoginAction, name="UserLoginAction"),	       
	       path("Register", views.Register, name="Register"),
	       path("RegisterAction", views.RegisterAction, name="RegisterAction"),	 
	       path("UploadMaterial", views.UploadMaterial, name="UploadMaterial"),
	       path("UploadMaterialAction", views.UploadMaterialAction, name="UploadMaterialAction"),
	       path("logout", views.logout, name="logout"),
	       path("TrainML", views.TrainML, name="TrainML"),
	       path("SearchTutorial", views.SearchTutorial, name="SearchTutorial"),
	       path("SearchTutorialAction", views.SearchTutorialAction, name="SearchTutorialAction"),	
	       path("DownloadDataAction", views.DownloadDataAction, name="DownloadDataAction"),
	       path("SearchTutorialVoice", views.SearchTutorialVoice, name="SearchTutorialVoice"),
	       path("SearchTutorialVoiceAction", views.SearchTutorialVoiceAction, name="SearchTutorialVoiceAction"),	
]