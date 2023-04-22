# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3. LabelEncoder and encode the dataset.
4. Import LogisticRegression from sklearn and apply the model on the dataset.
5. Predict the values of array.
6. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
7. Apply new unknown values

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Keerthika N
RegisterNumber: 212221230049
*/
```
```
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Removes the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")# A library for large linear classification
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)# Accuracy Score = (TP+TN)/
#accuracy_score(y_true,y_prednormalize=False)

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

```
## Output:
* Placement data

![1](https://user-images.githubusercontent.com/93427089/233767748-d82ac04d-b1d8-4501-a658-97ada3588b17.png)

* Salary data

![2](https://user-images.githubusercontent.com/93427089/233767753-862793ab-e99d-4523-940f-be48a97f3335.png)

* Checking the null() function

![3](https://user-images.githubusercontent.com/93427089/233767756-27987a8c-2af8-4f5e-afb0-7de7ddd99cae.png)

* Data Duplicate

![4](https://user-images.githubusercontent.com/93427089/233767767-9cc9a4d0-b3d0-4de0-b93a-05cdc52c6ef7.png)

* Print data

![5](https://user-images.githubusercontent.com/93427089/233767772-b3a85f9b-9e11-4100-8e6f-34a4a0f2e816.png)

* Data-status

![6i](https://user-images.githubusercontent.com/93427089/233767775-814fd1e7-defb-4efc-bd06-ebca3d439644.png)
![6ii](https://user-images.githubusercontent.com/93427089/233767779-16c9f2c3-b66c-4668-9eaa-9586ecd3eef9.png)

* y_prediction array

![7](https://user-images.githubusercontent.com/93427089/233767787-1e13eb09-232e-4c17-aa59-ceb8b3cbe00b.png)

* Accuracy value

![8](https://user-images.githubusercontent.com/93427089/233767791-acd1562a-cc87-47bc-85c7-003329291568.png)

* Confusion array

![9](https://user-images.githubusercontent.com/93427089/233767795-3a0fde62-ca9d-49d2-9116-cea25a633f76.png)

* Classification report

![10](https://user-images.githubusercontent.com/93427089/233767798-922f2df2-7044-450d-9be8-506888f972d6.png)

* Prediction of LR

![11](https://user-images.githubusercontent.com/93427089/233767809-d4c2cc04-539e-4232-b190-718e61d7cdaf.png)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
