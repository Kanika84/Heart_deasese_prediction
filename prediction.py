# Importing required libraries
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Loading and reading the dataset
heart = pd.read_csv("heart_cleveland_upload.csv")

# Creating a copy of the dataset so that it will not affect our original dataset
heart_df = heart.copy()

# Renaming some of the columns
heart_df = heart_df.rename(columns={'condition':'target'})
print(heart_df.head())

# Checking class distribution
print(heart_df['target'].value_counts())

# Splitting our data into X and y. Here y contains target data and X contains rest of the features
X = heart_df.drop(columns='target')
y = heart_df['target']

# Splitting our dataset into training and testing for this we will use train_test_split library
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaler = scaler.fit_transform(X_train)
X_test_scaler = scaler.transform(X_test)

# Creating and evaluating different classifiers
# Logistic Regression
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_scaler, y_train)
y_pred_log_reg = log_reg.predict(X_test_scaler)
print('Logistic Regression Accuracy: {}%\n'.format(round((accuracy_score(y_test, y_pred_log_reg) * 100), 2)))
print('Classification Report\n', classification_report(y_test, y_pred_log_reg))
print(confusion_matrix(y_test, y_pred_log_reg))

# Support Vector Machine
svc = SVC(random_state=42)
svc.fit(X_train_scaler, y_train)
y_pred_svc = svc.predict(X_test_scaler)
print('SVM Accuracy: {}%\n'.format(round((accuracy_score(y_test, y_pred_svc) * 100), 2)))
print('Classification Report\n', classification_report(y_test, y_pred_svc))
print(confusion_matrix(y_test, y_pred_svc))

# K-Nearest Neighbors
knn = KNeighborsClassifier()
knn.fit(X_train_scaler, y_train)
y_pred_knn = knn.predict(X_test_scaler)
print('KNN Accuracy: {}%\n'.format(round((accuracy_score(y_test, y_pred_knn) * 100), 2)))
print('Classification Report\n', classification_report(y_test, y_pred_knn))
print(confusion_matrix(y_test, y_pred_knn))

# Decision Tree Classifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train_scaler, y_train)
y_pred_dt = dt.predict(X_test_scaler)
print('Decision Tree Accuracy: {}%\n'.format(round((accuracy_score(y_test, y_pred_dt) * 100), 2)))
print('Classification Report\n', classification_report(y_test, y_pred_dt))
print(confusion_matrix(y_test, y_pred_dt))

# Random Forest Classifier
rf = RandomForestClassifier(n_estimators=20, random_state=42)
rf.fit(X_train_scaler, y_train)
y_pred_rf = rf.predict(X_test_scaler)
print('Random Forest Accuracy: {}%\n'.format(round((accuracy_score(y_test, y_pred_rf) * 100), 2)))
print('Classification Report\n', classification_report(y_test, y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf))

# Saving the best model 
filename = 'heart-disease-prediction-knn-model.pkl'
pickle.dump(rf, open(filename, 'wb'))
