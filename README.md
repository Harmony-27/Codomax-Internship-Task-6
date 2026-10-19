# Spam Message Classifier

## About the Project

This is a Machine Learning project that classifies SMS messages as spam or ham.

- Spam: Unwanted or suspicious messages.
- Ham: Normal messages.

The project uses the SMS Spam Collection dataset to train a Machine Learning model.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorization
- Multinomial Naive Bayes

## How It Works

1. Load the SMS dataset.
2. Remove duplicate messages.
3. Split the data into training and testing sets.
4. Convert text messages into numerical features using TF-IDF.
5. Train a Multinomial Naive Bayes model.
6. Evaluate the model using accuracy, precision, recall, F1-score, and a confusion matrix.
7. Allow users to enter messages and predict whether they are spam or ham.

## Model Performance

The model achieved approximately 95.26% accuracy on the test dataset.

The spam recall was approximately 63%, meaning the model did not identify every spam message. Model performance may vary on new messages.

## How to Run

1. Install Python.
2. Install the required libraries:

   pip install pandas scikit-learn

3. Download the SMS Spam Collection dataset from the UCI Machine Learning Repository.
4. Place the `SMSSpamCollection` file in the same folder as `spam_classifier.py`.
5. Run the program:

   python spam_classifier.py

6. Enter an SMS message to get a prediction.
7. Type `exit` to stop the program.

## Dataset

SMS Spam Collection Dataset  
Source: UCI Machine Learning Repository