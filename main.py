import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
warnings.filterwarnings("ignore")


#Setting up the training and test sets
training = pd.read_csv('training_set.csv')
test = pd.read_csv('test_set.csv')

train_original = training.copy()
test_original = test.copy()

#Change into numerical values
training['Loan_Status'].replace('Y', 1, inplace=True)
training['Loan_Status'].replace('N', 0, inplace=True)

'''Missing values and outlier treatment'''
print(training.isnull().sum())

#How to treat missing values:
# 1. For numerical variables: imputation using mean or median
# 2. For categorical variables: imputation using mode

training['Gender'].fillna(training['Gender'].mode()[0], inplace=True)
training['Married'].fillna(training['Married'].mode()[0], inplace=True)
training['Dependents'].fillna(training['Dependents'].mode()[0], inplace=True)
training['Self_Employed'].fillna(training['Self_Employed'].mode()[0], inplace=True)
training['Credit_History'].fillna(training['Credit_History'].mode()[0], inplace=True)

print(training['Loan_Amount_Term'].value_counts())
training['Loan_Amount_Term'].fillna(training['Loan_Amount_Term'].mode()[0], inplace=True)

training['LoanAmount'].fillna(training['LoanAmount'].median(), inplace=True)
print(training.isnull().sum())

test['Gender'].fillna(test['Gender'].mode()[0], inplace=True)
test['Married'].fillna(test['Married'].mode()[0], inplace=True)
test['Dependents'].fillna(test['Dependents'].mode()[0], inplace=True)
test['Self_Employed'].fillna(test['Self_Employed'].mode()[0], inplace=True)
test['Credit_History'].fillna(test['Credit_History'].mode()[0], inplace=True)

test['Loan_Amount_Term'].fillna(test['Loan_Amount_Term'].mode()[0], inplace=True)
test['LoanAmount'].fillna(test['LoanAmount'].median(), inplace=True)

#Outlier Treatment
'''E.g. Loan Amount -> Right skewness'''
plt.subplot(131)
training['LoanAmount'] = np.log(training['LoanAmount'])
test['LoanAmount'] = np.log1p(test['LoanAmount'])
training['LoanAmount'].hist(bins=20)
test['LoanAmount'] = np.log(test['LoanAmount'])

plt.subplot(132)
training['ApplicantIncome'] = np.log(training['ApplicantIncome'])
test['ApplicantIncome'] = np.log1p(test['ApplicantIncome'])
training['ApplicantIncome'].hist(bins=20)

plt.subplot(133)
training['CoapplicantIncome'] = np.log1p(training['CoapplicantIncome'])
test['CoapplicantIncome'] = np.log1p(test['CoapplicantIncome'])
training['CoapplicantIncome'].hist(bins=20)
# plt.show()

'''Model Building I'''
#Logistic Regression
training = training.drop(['Loan_ID'],axis=1)
test = test.drop(['Loan_ID'], axis=1)

X = training.drop(['Loan_Status'], axis=1)
y = training.Loan_Status


#Make Dummy variables
X = pd.get_dummies(X)
training = pd.get_dummies(training)
test = pd.get_dummies(test)


#Create validation sets
x_train, x_cv, y_train, y_cv = train_test_split(X, y, test_size=0.3)

#Train model
model = LogisticRegression()
model.fit(x_train, y_train)
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True, intercept_scaling=1,
                   max_iter=100, multi_class='ovr', n_jobs=1, penalty='l2', random_state=1, solver='liblinear',
                   tol=0.0001, verbose=0, warm_start=False)

predictions_training = model.predict(x_cv)

print("Accuracy:" + str(accuracy_score(y_cv, predictions_training)))
print("Precision:" + str(precision_score(y_cv, predictions_training)))
print("Recall:" + str(recall_score(y_cv, predictions_training)) + "\n")


predictions_test = model.predict(test)
result = test_original.copy()
result['Loan_Status'] = predictions_test
result['Loan_Status'].replace({1: 'Y', 0: 'N'}, inplace=True)
result.to_csv("logistics.csv")
