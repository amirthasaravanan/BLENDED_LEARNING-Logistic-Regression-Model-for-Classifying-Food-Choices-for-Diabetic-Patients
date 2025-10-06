# BLENDED_LEARNING
# Implementation of Logistic Regression Model for Classifying Food Choices for Diabetic Patients

## AIM:
To implement a logistic regression model to classify food items for diabetic patients based on nutrition information.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import necessary libraries.
2. Load the dataset using pd.read_csv().
3. Display data types, basic statistics, and class distributions.
4. Visualize class distributions with a bar plot.
5. Scale feature columns using MinMaxScaler.
6. Encode target labels with LabelEncoder.
7. Split data into training and testing sets with train_test_split().
8. Train LogisticRegression with specified hyperparameters and evaluate the model using metrics and a confusion matrix plot. 

## Program:
```
Program to implement Logistic Regression for classifying food choices based on nutritional information.
Developed by: AMIRTHA VARSHINI M
RegisterNumber:  212224230017
```
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
```
```python

#Load the dataset
df = pd.read_csv('food_items (1).csv')
```
```python
# Inspect the dataset
print('Name: AMIRTHA VARSHINI M')
print('Reg. No: 212224230017 ')
print("Dataset Overview:")
print(df.head())

print("\nDataset Info:")
print(df.info())

X_raw = df.iloc[:, :-1]
y_raw = df.iloc[:, -1:]

scaler = MinMaxScaler()
```
```python
# Scaling the raw input features
X = scaler.fit_transform(X_raw)
```
```python
# Create a LabelEncoder object
label_encoder = LabelEncoder()
```
```python
# Encode the target variable
y = label_encoder.fit_transform(y_raw.values.ravel())
```
```python
# First, let's split the training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=123)
```
```python
# L2 penalty to shrink coefficients without removing any features from the model
penalty = 'l2'
```
```python
# Our classification problem is multinomial
multi_class = 'multinomial'

# Use lbfgs for L2 penalty and multinomial classes
solver = 'lbfgs'

# Max iteration = 1000
max_iter = 1000

# Define a logistic regression model with above arguments
l2_model = LogisticRegression(random_state=123, penalty=penalty, multi_class=multi_class, solver=solver, max_iter=max_iter)

l2_model.fit(X_train, y_train)

y_pred = l2_model.predict(X_test)
```
```python
# Evaluate the model
print('Name: AMIRTHA VARSHINI M ')
print('Reg. No: 212224230017 ')
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```
```python
# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

print('Name: AMIRTHA VARSHINI M ')
print('Reg. No: 212224230017 ')
```

## Output:
### Inspect the dataset
![Untitled10 i… (6) - JupyterLab_removed_page-0001](https://github.com/user-attachments/assets/b3f0f867-7f2e-4448-947f-a082709fc395)


### Evaluate the dataset
<img width="1348" height="376" alt="Screenshot 2025-10-06 172323" src="https://github.com/user-attachments/assets/85aec96f-5d56-40de-962f-6eabe98800f0" />

### Confusion matrix
<img width="1349" height="120" alt="Screenshot 2025-10-06 172526" src="https://github.com/user-attachments/assets/672b319c-ab90-4e94-aaeb-7454880ff41a" />


## Result:
Thus, the logistic regression model was successfully implemented to classify food items for diabetic patients based on nutritional information, and the model's performance was evaluated using various performance metrics such as accuracy, precision, and recall.
