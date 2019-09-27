# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 06:55:01 2019

@author: Hitesh
"""


""""Hackathon Loan Dataset."""

"""Importing Library"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

loan_df=pd.read_csv("C:\\Users\\Hitesh\\Downloads\\Loan Hacklethon\\train.csv")
print(loan_df.head(5))

"""EDA od Loan Data"""
print(loan_df.columns)
print(loan_df.shape)
loan_df.isnull().sum()

"""Splitting dataset into Independent and Dependent data."""
x_data=loan_df.iloc[:,0:28]
y_data=loan_df.iloc[:,-1]

x_data.head(10)
y_data.head(5)

"""Performing Preprocessing Dummie Encoding technique."""
dummies=pd.get_dummies(x_data['source'])
dummies.head(5)

merge=pd.concat([x_data,dummies],axis='columns')
merge.head(5)

loan_data=merge.drop(['source','Z'],axis='columns')
loan_data.head(5)

#print(loan_data['number_of_borrowers'])

"""dropping financial_institution,origination_date,first_payment_date and loan_purpose columns."""
loan_data=loan_data.drop(['insurance_type','m1','number_of_borrowers','debt_to_income_ratio','borrower_credit_score','interest_rate','financial_institution','origination_date','first_payment_date','loan_purpose'],axis=1)
loan_data.head(5)

"""Pipeline for best Feature Selection."""
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

"""Applying SelectKBest class to select top 15 feature"""
bestfeature=SelectKBest(score_func=chi2,k=20)
fit=bestfeature.fit(loan_data,y_data)

dfscore=pd.DataFrame(fit.scores_)
dfcolumn=pd.DataFrame(loan_data.columns)

#Combining two columns.
featurescore=pd.concat([dfcolumn,dfscore],axis=1)
featurescore.columns=['cols_name','Score']
featurescore

#Selecting top 15 feature from Features who having larget values.
print(featurescore.nlargest(18,'Score'))

"""Dropping Remaining columns."""
loan_info=loan_data.drop(['X','Y'],axis=1)
loan_info.head(5)

print(loan_info.columns)

"""Count 0 value in dataset."""
print((loan_info[['loan_to_value','loan_term','insurance_percent','loan_id','co-borrower_credit_score','m2','m3','m4','m5','m6','m7','m8','m9','m10','m11','m12']] == 0).sum())

"""Mark zero value as NaN value"""
loan_info[['loan_to_value','loan_term','insurance_percent','loan_id','co-borrower_credit_score','m2','m3','m4','m5','m6','m7','m8','m9','m10','m11','m12']] = loan_info[['loan_to_value','loan_term','insurance_percent','loan_id','co-borrower_credit_score','m2','m3','m4','m5','m6','m7','m8','m9','m10','m11','m12']].replace(0, np.NaN)
print(loan_info.head(5))

print(loan_info.isnull().sum())

#Replacing NaN values with it's mean() value.
loan_info.fillna(loan_info.mean(),inplace=True)

loan_info.head(5)
print(loan_info.isnull().sum())

y_data.head(10)

"""Splitting data into training and testing data."""
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(loan_info,y_data,test_size=0.25,random_state=40)

print(x_train.size)
print(y_train.size)
print(x_test.size)
print(y_test.size)

"""Feature Scalling Pipeline."""
#Doing feature scalling on loan_id,unpaid_principal_bal and borrower_credit_score columns.
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)
print(X_train)
print(X_test)

"""Selecting Different algorithms."""
#Selecting Logistic Regression Algorithm.
from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression(random_state=0)

#Selecting RandomForest Algorithm.
from sklearn.ensemble import RandomForestClassifier
rand_f=RandomForestClassifier(n_estimators=100,random_state=0)

#Selecting KNN Algorithm.
from sklearn.neighbors import KNeighborsClassifier
knearest=KNeighborsClassifier(n_neighbors=5)

#Selecting Decision tree Algorithm.
from sklearn.tree import DecisionTreeClassifier
tree_model=DecisionTreeClassifier()

"""Cross Validation technique."""
"""Importing cross validation library ."""
"""Checking accuracy for Logistic Regression Algorithm."""
from sklearn.model_selection import cross_val_score
cross_val_score(logreg,loan_info,y_data,cv=5,scoring='accuracy').mean()

"""Checking accuracy for Random Forest Algorithm."""
cross_val_score(rand_f,loan_info,y_data,cv=5,scoring='accuracy').mean()

"""Checking accuracy for K-Nearest Neighbour Algorithm."""
cross_val_score(knearest,loan_info,y_data,cv=5,scoring='accuracy').mean()

"""Cheking score for Decision tree Algorithm."""
cross_val_score(tree_model,loan_info,y_data,cv=5,scoring='accuracy').mean()

"""Ramdom Forest Algorithm gives us good accuracy So we use Random Forest."""
"""Fitting data into random forest Algorithm."""
rand_f.fit(X_train,y_train)

"""Making Prediction of loan."""
y_pred=rand_f.predict(X_test)
print(y_pred)

"""Caculating Acuuracy and F-1 score."""
from sklearn.metrics import confusion_matrix,f1_score,accuracy_score
f_score=f1_score(y_test,y_pred)
print("F1-Score is =",f_score)

print('\n')

cm=confusion_matrix(y_test,y_pred)
print("Confusion Matrix = ",cm)

score=accuracy_score(y_test,y_pred)
print(score)

#Testing Model By applying Test data.
"""Testing Model by applying Test Data."""
test_data=pd.read_csv("C:\\Users\\Hitesh\\Downloads\\Loan Hacklethon\\test.csv")
test_data.head(5)

print(test_data.shape)
print(test_data.columns)

"""Removing Unwanted column from Test Data."""
Test_data=test_data.drop(['insurance_type','m1','number_of_borrowers','debt_to_income_ratio','borrower_credit_score','interest_rate','financial_institution','origination_date','first_payment_date','loan_purpose','source'],axis=1)
Test_data.head(5)

print(Test_data.isnull().sum())

"""Feature Scalling Pipeline."""
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
Test_data_info=sc.fit_transform(Test_data)
print(Test_data_info)

"""Making Prediction on Test Data."""
y_pred_data=rand_f.predict(Test_data_info)
print(y_pred_data)

pred_data=pd.DataFrame(y_pred_data)
pred_data.head(5)

pred_data['m13']=pd.DataFrame(y_pred_data)
pred_data.head(5)

print(Test_data.head(5))

"""Merge test data and output column"""
Merge_data=pd.concat([Test_data,pred_data],axis='columns')
Merge_data.head(5)

Merge_data.drop([0],axis=1,inplace=True)
Merge_data.head(5)

"""Drop all unwanted columns."""
Merge_data.drop(['unpaid_principal_bal','loan_to_value','loan_term','insurance_percent','co-borrower_credit_score','m2','m3','m4','m5','m6','m7','m8','m9','m10','m11','m12'],axis=1,inplace=True)

print(Merge_data.head(5))

"""Converting Output file in csv file."""
Merge_data.to_csv("Final_Loan_Code",index=False)

print(y_pred_data.size)
print(y_test.size)
print(y_pred.size)
print(y_test.size)
#creating random value.
data=np.random.random_integers(0,1,6851)
print(data)
type(data)

#Converting into series.
result=pd.Series(data)

#Concat two Series.
data1=pd.concat([y_test,result])

print(data1)

#Renaming name
y_test=data1.rename()
y_test
print(y_test.size)

from sklearn.metrics import f1_score,accuracy_score
fc=f1_score(y_test,y_pred_data)
print(fc)
score=accuracy_score(y_test,y_pred_data)
print(score)













