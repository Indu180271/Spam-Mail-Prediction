# Spam Mail Prediction Using Machine Learning
![image alt](https://github.com/Indu180271/Spam-Mail-Prediction/blob/d3d89739964d0d2894861b77cd40992736d9b992/Spam_mail_prediction/Sample_Output.png)
## Introduction
In today's world, email is a key method of communication for personal, business, corporate, and government purposes. With the rise in email usage, there has been a notable increase in spam emails. These spam emails, also known as junk emails, are mass-sent messages that can be both annoying and pose security risks to computer systems.
The aim of this project is to develop a model that uses ML Algorithms to accurately predict the spam emails.
## Dataset
The dataset used for this project is a labelled data.The shape of the Dataset is (5572, 2).<br>
Data Dictionary:<br>
Category ----> type(Categorical) ----> Label indicating whether the message is 'ham' (not spam) or 'spam'.<br>
Message ----> type(Text) ----> The content of the SMS message.
## Methodology
1. Data Collection: Obtain a labelled dataset containing emails classified as "spam" and "ham" (non-spam) from Kaggle.
2. Data Preprocessing: Clean and preprocess the text data by removing any irrelevant information such as special characters, stop words, and converting text to lowercase to standardize it. TfidfVectorizer to convert the text data into numerical features that can be used by the machine learning model. The TF-IDF vectorizer transforms the text into feature vectors representing the importance of each word in the context of the dataset. Encode the labels where "ham" is encoded as 0 and "spam" is encoded as 1 using LabelEncoding.
3. Split the data into training and testing sets.
5. Model Building: Train a Logistic Regression, KNN models using the training data.
6. Model Evaluation: Evaluate the model's performance using accuracy, precision, recall, and F1 score metrics. These metrics will help assess how well the model distinguishes between spam and ham emails.
7. Model Saving: Save the trained model and the vectorizer using libraries like pickle or joblib so they can be loaded later for making predictions.
8. Model Deployment: Using Streamlit Application

## Results
Logistic Regression <br>
Accuracy on training data :  0.9676912721561588<br>
Accuracy on test data :  0.9668161434977578<br>
KNN<br>
Accuracy on training data: 0.9201256450527261<br>
Accuracy on test data: 0.9094170403587444<br>
