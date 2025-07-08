#include <stdio.h>

// Define the maximum size for the stacks
#define STACK_SIZE 100

// Declare two stacks using arrays
int stack1[STACK_SIZE];
int stack2[STACK_SIZE];
int top1 = -1;  // Top index for stack1
int top2 = -1;  // Top index for stack2

// Function to check if stack1 is empty
int isEmptyStack1() 
{
    return top1 == -1;
}

// Function to check if stack2 is empty
int isEmptyStack2() 
{
    return top2 == -1;
}

// Function to check if stack1 is full
int isFullStack1() 
{
    return top1 == STACK_SIZE - 1;
}



// Function to check if stack2 is full
int isFullStack2() 
{
    return top2 == STACK_SIZE - 1;
}








// Function to push an element onto stack1
void pushStack1(int value) 
{
    if (isFullStack1())
    {
        printf("Stack1 overflow\n");
        return;
    }
    
   stack1[++top1] = value;  // Increment top1 and store the value in stack1
}

// Function to push an element onto stack2
void pushStack2(int value) 
{
    if (isFullStack2()) 
     {
        printf("Stack2 overflow\n");
        return;
    }
    stack2[++top2] = value;  // Increment top2 and store the value in stack2
}
	
// Function to pop an element from stack1
int popStack1() 
{
    if (isEmptyStack1()) 
     {
        printf("Stack1 underflow\n");
        return -1;  // Return -1 in case of stack1 underflow
    }
    return stack1[top1--];  // Return the top element of stack1 and decrement top1
}





// Function to pop an element from stack2
int popStack2() 
{
    if (isEmptyStack2())
    {
        printf("Stack2 underflow\n");
        return -1;  // Return -1 in case of stack2 underflow
    }
    return stack2[top2--];  // Return the top element of stack2 and decrement top2
}

// Function to enqueue an element into the queue
void enqueue(int value) 
{
    pushStack1(value);  // Simply push the element onto stack1
    printf("Enqueued %d\n", value);
}

// Function to dequeue an element from the queue
int dequeue() 
{
    // If both stacks are empty, the queue is empty
    if (isEmptyStack1() && isEmptyStack2())
     {
        printf("Queue is empty\n");
        return -1;
    }
  
  // If stack2 is empty, transfer all elements from stack1 to stack2
    if (isEmptyStack2())  
    {
        while (!isEmptyStack1()) 
        {
            pushStack2(popStack1());  // Move elements from stack1 to stack2
        }
    }

    // Pop the top element from stack2 (this is the front of the queue)
    int dequeuedValue = popStack2();
    printf("Dequeued %d\n", dequeuedValue);
    return dequeuedValue;
}

// Main function to demonstrate the working of the queue
int main() 
{
    enqueue(10);
    enqueue(20);
    enqueue(30);

    dequeue();
    dequeue();

    enqueue(40);
    dequeue();
    dequeue();


    return 0;
}

