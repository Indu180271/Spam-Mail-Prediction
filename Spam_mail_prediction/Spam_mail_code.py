# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 16:27:38 2024

@author: M INDHU
"""
'''
# CRISP - ML(Q):
    1) Business and data understanding
    2) Data preparation
    3) Model Building & Tuning
    4) Evaluating Model
    5) Deployment
    6) Business Monitoring and Maintenance
    
Problem Statement:
The goal is to develop a machine learning model that can accurately classify SMS messages as either "spam" or "ham" (non-spam).
This will help telecommunications companies, mobile service providers, and messaging apps to effectively filter and 
block spam messages, ensuring that users receive only legitimate and relevant messages.

Data Dictionary:
Category	    type(Categorical)	       Label indicating whether the message is 'ham' (not spam) or 'spam'.
Message	        type(Text)         	       The content of the SMS message.  
 
'''    


# importing the necessary libraries

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report



# Loading Dataset
mail_data = pd.read_csv(r"C:\Users\M INDHU\Documents\Clg_projects\mail_data.csv")

# checking the number of rows and columns in the dataframe
mail_data.shape


# print first 5 rows of dataset
mail_data.head(5)

# Check for missing values
mail_data.columns
mail_data['Message'].isna().sum()   # No missing values found in data set


# Encode the Labels apply on category(output feature)
label_encoder = LabelEncoder()
mail_data['Category'] = label_encoder.fit_transform(mail_data['Category'])

# print the lebels   [ham  -- 0, spam ---- 1]
print(label_encoder.classes_)


# Split the data into features and target 
X = mail_data['Message'] 
Y = mail_data['Category']

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 3)
  

# Vectorize the text data
vectorizer = TfidfVectorizer(min_df = 1, stop_words = 'english', lowercase = True)
X_train_features = vectorizer.fit_transform(X_train)   # fit_transform This method is used to both fit the model (learn from the data) and transform the data based on the learned parameters.
X_test_features = vectorizer.transform(X_test)  # transform do not learn from the data



# print the shape of x, X_train, X_test
print(X.shape)   # (5572,)
print(X_train.shape)    # (4457,)
print(X_test.shape)     # (1115,)


# Model Building 
# Logistic Regressin
model = LogisticRegression()

# tarin the logisticRegression model with the training model
model.fit(X_train_features, Y_train)


# prediction on training data
prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)
print('Accuracy on training data : ', accuracy_on_training_data)


# prediction on testing data
prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)
print('Accuracy on test data : ', accuracy_on_test_data)





# Model Building
# K - Nearest Neighbors Algorithms
model_1 = KNeighborsClassifier(n_neighbors=5)  # You can change the number of neighbors

# Train the k-NN model with the training data
model_1.fit(X_train_features, Y_train)

# Prediction on training data
prediction_on_training_data = model_1.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)

print('Accuracy on training data:', accuracy_on_training_data)

# Prediction on test data
prediction_on_test_data = model_1.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)
print('Accuracy on test data:', accuracy_on_test_data)

# Print classification report
print(classification_report(Y_test, prediction_on_test_data))

"""
Among 2 Models Logistic Regression Performs well
1. Logistic Regression 
Accuracy on training data :  0.9676912721561588
Accuracy on test data :  0.9668161434977578

2. KNN
Accuracy on training data: 0.9201256450527261
Accuracy on test data: 0.9094170403587444

"""


# Saving the Best model
# Train the logistic regression model with the training data
model.fit(X_train_features, Y_train)

# Save the trained model to a file
import pickle

# Save the model to a file
with open('logistic_regression_model.pkl', 'wb') as file:
    pickle.dump(model, file)



from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Create and fit the vectorizer on the training data
vectorizer = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_features = vectorizer.fit_transform(X_train)

# Save the vectorizer to a file
joblib.dump(vectorizer, 'vectorizer.pkl')






















import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Combine all messages into a single string
all_messages = ' '.join(mail_data['Message'])

# Generate the word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_messages)

# Plot the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # Hide axes
plt.show()






















