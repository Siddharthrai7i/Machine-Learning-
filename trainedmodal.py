import pandas as pd 
import numpy as np
import seaborn as sns 


df=pd.read_csv('diabetes.csv')

X=df.iloc[:,:-1]
y=df.iloc[:,-1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)


from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()

from sklearn.model_selection import GridSearchCV
parameter={
    'penalty':['l1','elsticnet' ,'l2'],
    'C':[1,2,3,4,5,6,10,20,30,40,50] ,
    'max_iter':[100,200,300]}

classifier_regressor=GridSearchCV(classifier ,param_grid=parameter,scoring='accuracy',cv=5)

classifier_regressor.fit(X_train,y_train)

#print(classifier_regressor.best_params_)
#print(classifier_regressor.best_score_)

y_pred=classifier_regressor.predict(X_test)

y_pred=classifier_regressor.predict(X_test)
from sklearn.metrics import accuracy_score,classification_report
score=accuracy_score(y_pred , y_test)

import pickle as pk

with open('classifier.pkl' , 'wb') as file :
    pk.dump(classifier ,file)
