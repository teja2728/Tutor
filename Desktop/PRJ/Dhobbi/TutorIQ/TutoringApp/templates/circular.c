#include <stdio.h>
#include <stdlib.h>





// Definition of a node in the circular linked list
struct Node {
    int data;
    struct Node* next;
};

// Function to create a new node
struct Node* createNode(int data) {
    struct Node* newNode = (struct Node*)malloc(sizeof(struct Node));
    newNode->data = data;
    newNode->next = NULL;
    return newNode;
}

// Function to create a circular linked list with n people
struct Node* createCircularList(int n) {
    struct Node* head = createNode(1);  // First person
    struct Node* prev = head;

    // Create the rest of the nodes (people)
    for (int i = 2; i <= n; i++) {
        struct Node* newNode = createNode(i);
        prev->next = newNode;
        prev = newNode;
    }

    // Make the list circular by connecting the last node to the head
    prev->next = head;

    return head;
}

// Function to solve the Josephus problem using a circular linked list
int josephus(int n) {
    // Create a circular linked list with n people
    struct Node* head = createCircularList(n);
    struct Node* prev = NULL;
    struct Node* current = head;

    // Continue until only one person remains
    while (current->next != current) {
        // Move to the next person (current has the sword)
        prev = current;
        current = current->next;

        // Remove the person that current points to
        prev->next = current->next;
        printf("Person %d is killed.\n", current->data);

        // Free the memory of the killed person
        free(current);

        // Move the current pointer to the next person
        current = prev->next;
    }

    // The last person remaining
    int luckyPerson = current->data;

    // Free the last remaining node
    free(current);

    return luckyPerson;
}

// Main function to test the josephus function
int main() {
    int n;
    printf("Enter the number of people in the circle: ");
    scanf("%d", &n);

    int luckyPerson = josephus(n);
    printf("The luckiest person is at position: %d\n", luckyPerson);

    return 0;
}
