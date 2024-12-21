import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,accuracy_score
import pickle

# Load the dataset
data=pd.read_csv(r"D:\VS_CODE(PROJECTS-NARESH-IT)\LOAN APPROVAL PREDICTION\test_case\Loan_Data.csv")
en=LabelEncoder()
data['Property_Area'] = en.fit_transform(data['Property_Area'])
data['Loan_Status']=en.fit_transform(data['Loan_Status'])
x = data.iloc[:, [6, 10,11]]
y = data.iloc[:,-1]

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=.2)

classifier=RandomForestClassifier(random_state =50)
classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)

cm=confusion_matrix(y_test,y_pred)
print(cm)

acc=accuracy_score(y_test,y_pred)
print(acc)

pickle.dump(classifier,open('random_foresr','wb'))