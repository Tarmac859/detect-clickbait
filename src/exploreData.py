#%%
import pandas as pd
import numpy as np
import seaborn as sns
import string
import nltk
import os
import re
from matplotlib import pyplot as plt
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from gensim.models import KeyedVectors

#%%
def predict_clickbait_svm(title):
    preprocessed_title = preprocess_text(title)
    title_embedding = get_average_embedding(preprocessed_title)
    title_embedding = np.reshape(title_embedding, (1, -1))  # Reshape to match SVM input
    prediction = svm_model.predict(title_embedding)[0]
    return prediction
# %%
# Naive Bayes Model Usage
def predict_clickbait(title):
    preprocessed_title = preprocess_text(title)
    X_new = vectorizer.transform([preprocessed_title])
    prediction = model.predict(X_new)
    return prediction[0]

# %%
# Predict given title for linear regression function
def predict_clickbait_linear_regression(title):
    preprocessed_title = preprocess_text(title)
    tfidf_vector = vectorizer.transform([preprocessed_title])
    clickbait_probability = model1.predict(tfidf_vector)[0]
    clickbait_label = 1 if clickbait_probability >= 0.5 else 0
    return clickbait_label, clickbait_probability

#%%
def get_average_embedding(title):
    tokens = title.split()
    embeddings = [word_vectors[word] for word in tokens if word in word_vectors]
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(300)
#%%
"""def preprocess_text(text): # base
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in stopwords.words('english')]
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text"""
    
def preprocess_text(text): # With numbers
    text = text.lower()
    
    # Remove punctuation, except for hyphen
    punctuation_to_remove = string.punctuation.replace('-', '')
    text = text.translate(str.maketrans('', '', punctuation_to_remove))
    
    # Preserve numbers using regex pattern
    numbers_pattern = r"\b\d+\b"  # Matches any sequence of digits (numbers)
    text = re.sub(numbers_pattern, 'NUM', text)
    
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in stopwords.words('english')]
    
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

#%%
filename = r'C:\Users\tmand\Documents\Projects\Demo\Data\clickbait_data.csv'
glove_file = r'C:\Users\tmand\Documents\Projects\Demo\Data\GoogleNews-vectors-negative300.bin'

word_vectors = KeyedVectors.load_word2vec_format(glove_file, binary=True)

df = pd.read_csv(filename, header=None, names=['headline', 'clickbait_label'])
df['preprocessed_title'] = df['headline'].apply(preprocess_text)
df['glove_embedding'] = df['preprocessed_title'].apply(get_average_embedding)


#%%
#Feature extraction
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['preprocessed_title'])
y = df['clickbait_label']

# %%
#Split data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
# Training and Evaluation for naive bayes
model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

# %%
# Example usage
title = "Learn how to cook like a chef with these easy kitchen hacks!"
prediction = predict_clickbait(title)
print(f"Prediction: {prediction}")

#%%
# Linear regression Train and predict 
model1 = LinearRegression()
model1.fit(X_train, y_train)
y_pred1 = model1.predict(X_test)

#%%
# Convert predicted probiabilities to binary labels (clickbait or non clickbait)
yPredBinary = [1 if prob >= 0.5 else 0 for prob in y_pred1]

#%%
# Calculate accuracy for linear regression
accuracy = accuracy_score(y_test, yPredBinary)
print("Accuracy:", accuracy)

# %% Linear regression usage
title = "Learn how to cook like a chef with these easy kitchen hacks!"
prediction, probability = predict_clickbait_linear_regression(title)
print(f"Prediction: {prediction}")

# %%
# Logistic regression
X1 = np.stack(df['glove_embedding'])
y1 = df['clickbait_label']

X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.2, random_state=42)

logReg_model = LogisticRegression()
logReg_model.fit(X_train1, y_train1)

y_pred2 = logReg_model.predict(X_test1)

logistic_regression_accuracy = accuracy_score(y_test1, y_pred2)
classification_report_str = classification_report(y_test1, y_pred2)

print("Accuracy:", logistic_regression_accuracy)
print("Classification Report:")
print(classification_report_str)


# %%
# Using the trainig line from before Train an SVM model
svm_model = SVC(kernel='linear', C=1.0)
svm_model.fit(X_train1, y_train1)

# Evaluate the SVM model
y_pred3 = svm_model.predict(X_test1)

print("Accuracy:", accuracy_score(y_test1, y_pred3))
print("Classification Report:")
print(classification_report(y_test1, y_pred3))
print("Confusion Matrix:")
print(confusion_matrix(y_test1, y_pred3))


#%%
title = "New Study Reveals Insights into Climate Change Patterns"
svm_prediction = predict_clickbait_svm(title)
print("SVM Prediction:", svm_prediction)


#knearest neighbors (stock price)

# %%
