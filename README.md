# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the placement dataset and remove unnecessary columns and duplicate values.
2.Convert categorical data into numerical values using Label Encoding.
3.Split the dataset into training and testing sets, then train the Logistic Regression model.
4.Predict the results, calculate accuracy, and display the confusion matrix and classification report. 

## Program:
```

Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: AKASH A
RegisterNumber:212225230008  

import pandas as pd
import numpy as np
df=pd.read_csv(r"C:\Users\acer\Downloads\Placement_Data.csv")
df
df1=df.copy()
df1
df1=df1.drop(['sl_no','salary'],axis=1)
df1.isnull().sum()
df1.duplicated().sum()
df1
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df1['gender']=le.fit_transform(df1['gender'])
df1['ssc_b']=le.fit_transform(df1['ssc_b'])
df1['hsc_b']=le.fit_transform(df1['hsc_b'])
df1['hsc_s']=le.fit_transform(df1['hsc_s'])
df1['degree_t']=le.fit_transform(df1['degree_t'])
df1['workex']=le.fit_transform(df1['workex'])
df1['specialisation']=le.fit_transform(df1['specialisation'])
df1['status']=le.fit_transform(df1['status'])
df1
x=df1.iloc[:,:-1]
x
y=df1['status']
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(solver="liblinear")
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy=accuracy_score(y_test,y_pred)
confusion=confusion_matrix(y_test,y_pred)
cr=classification_report(y_test,y_pred)
print("Accuracy score:",accuracy)
print("\nConfusion matrix:\n",confusion)
print("\nClassification Report:\n",cr)
from sklearn import metrics
cn_display=metrics.ConfusionMatrixDisplay(confusion_matrix=confusion,display_labels=['true','false'])
cn_display.plot()


```

## Output:


<img width="1262" height="532" alt="589377379-da9a4e5b-7a83-497c-8380-5fddbec9c3be" src="https://github.com/user-attachments/assets/5c8f15cb-07af-452a-bf06-7d466a7f7ef0" />
<img width="1251" height="676" alt="589377648-5738129e-8a38-4097-9792-cd2d6f804792" src="https://github.com/user-attachments/assets/ea89e5a9-f13c-46cb-bb63-7a8350936857" />



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
