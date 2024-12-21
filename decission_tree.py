import pandas as pd
import numpy as np 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pickle

data=pd.read_csv(r"D:\VS_CODE(PROJECTS-NARESH-IT)\LOAN APPROVAL PREDICTION\test_case\Loan_Data.csv")

lb=LabelEncoder()
data["Loan_Status"]=lb.fit_transform(data["Loan_Status"])
data["Property_Area"]=lb.fit_transform(data["Property_Area"])


x=data.iloc[:,[6,10,11]]
y=data.iloc[:,-1]
print(x)

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=.2)

classifier=DecisionTreeClassifier()
classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)
print(y_pred)
print(y_test)
ac=accuracy_score(y_test,y_pred)
print(ac)

pickle.dump(classifier,open("decission_tress.pkl","wb"))
