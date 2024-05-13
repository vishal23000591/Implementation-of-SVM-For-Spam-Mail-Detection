# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages.
2. Import the dataset to operate on.
3. Split the dataset.
4. Predict the required output.
5. End the program.


## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Vishal S
RegisterNumber:  212223110063
*/
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
df=pd.read_csv("letter-recognition.csv")
df.head()
df.isnull().sum()
df.info()
x=df.iloc[:,1:].values
y=df.iloc[:,0].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)
model=SVC()
model
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
y_pred
score=accuracy_score(y_test,y_pred)
print(score)
```

## Output:
# Data Head:
![ml 1](https://github.com/vishal23000591/Implementation-of-SVM-For-Spam-Mail-Detection/assets/147139719/3f106c29-a7a9-4f81-a997-f99e71b63d44)
# Data Info:
![ml 3](https://github.com/vishal23000591/Implementation-of-SVM-For-Spam-Mail-Detection/assets/147139719/1eb42fb4-d7c8-4bb2-a7dd-86ccfa69bdb5)
# Data is.null():
![ml 2](https://github.com/vishal23000591/Implementation-of-SVM-For-Spam-Mail-Detection/assets/147139719/01fe0ec5-c3c5-4eef-ae3c-78833b92b8b2)
# y_pred:
![ml 4](https://github.com/vishal23000591/Implementation-of-SVM-For-Spam-Mail-Detection/assets/147139719/da84180e-7b85-43b4-a976-359dd250b5af)
# Accuracy:
![ml 5](https://github.com/vishal23000591/Implementation-of-SVM-For-Spam-Mail-Detection/assets/147139719/39a003ca-cbb4-4d32-a45a-4de0e76858fd)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
