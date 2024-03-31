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
# print(training.isnull().sum())

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
# print(training.isnull().sum())

test['Gender'].fillna(test['Gender'].mode()[0], inplace=True)
test['Married'].fillna(test['Married'].mode()[0], inplace=True)
test['Dependents'].fillna(test['Dependents'].mode()[0], inplace=True)
test['Self_Employed'].fillna(test['Self_Employed'].mode()[0], inplace=True)
test['Credit_History'].fillna(test['Credit_History'].mode()[0], inplace=True)

test['Loan_Amount_Term'].fillna(test['Loan_Amount_Term'].mode()[0], inplace=True)
test['LoanAmount'].fillna(test['LoanAmount'].median(), inplace=True)

#Outlier Treatment
'''E.g. Loan Amount -> Right skewness'''
# plt.subplot(131)
# training['LoanAmount'] = np.log(training['LoanAmount'])
# test['LoanAmount'] = np.log1p(test['LoanAmount'])
# training['LoanAmount'].hist(bins=20)
# test['LoanAmount'] = np.log(test['LoanAmount'])
#
# plt.subplot(132)
# training['ApplicantIncome'] = np.log(training['ApplicantIncome'])
# test['ApplicantIncome'] = np.log1p(test['ApplicantIncome'])
# training['ApplicantIncome'].hist(bins=20)
#
# plt.subplot(133)
# training['CoapplicantIncome'] = np.log1p(training['CoapplicantIncome'])
# test['CoapplicantIncome'] = np.log1p(test['CoapplicantIncome'])
# training['CoapplicantIncome'].hist(bins=20)
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

#Model robustness using Validation
#Validation set, k-fold cross validation, Leave one out cross validation, and stratified k-fold cross validation
#We'll be using Stratified k-fold validation

'''Stratification is the process of rearranging the data so as to ensure that each fold is a good representative of
the whole'''

from sklearn.model_selection import StratifiedKFold

i = 1
kf = StratifiedKFold(n_splits=5, random_state=1,shuffle=True)

for train_index, test_index in kf.split(X, y):
    print(f'\n{i} of kfold {kf.n_splits}')
    xtr, xvl = X.iloc[train_index], X.iloc[test_index]
    ytr, yvl = y.iloc[train_index], y.iloc[test_index]
    model = LogisticRegression(random_state=1)
    model.fit(xtr, ytr)
    pred_test = model.predict(xvl)
    score = accuracy_score(yvl, pred_test)
    print('accuracy score', score)
    i += 1
    pred_test = model.predict(test)
    pred = model.predict_proba(xvl)[:, 1]

#Feature engineering (new features that can affect the target variable)

''' Total income (Applicant + Co-Applicant), EMI (amount to be paid per month), Balance income (Income left after EMI)'''

#Total Income
training['TotalIncome'] = training['ApplicantIncome'] + training['CoapplicantIncome']
test['TotalIncome'] = test['ApplicantIncome'] + test['CoapplicantIncome']

import seaborn as sns
# plt.figure(2)
# sns.distplot(training['TotalIncome'])
# plt.show()

#Right Skewed, thus we take the log transformation

training['TotalIncome'] = np.log(training['TotalIncome'])
test['TotalIncome'] = np.log(test['TotalIncome'])
plt.figure(3)
sns.distplot(training['TotalIncome'])
# plt.show()

#EMI
training['EMI'] = training['LoanAmount']/training['Loan_Amount_Term']
test['EMI'] = test['LoanAmount']/test['Loan_Amount_Term']
plt.figure(4)
sns.distplot(training['EMI'])
# plt.show()
training['EMI'] = np.log(training['EMI'])
test['EMI'] = np.log(test['EMI'])

#Balance Income
training['BalanceIncome'] = training['TotalIncome'] - training['EMI']
test['BalanceIncome'] = test['TotalIncome'] - test['EMI']
plt.figure(5)
sns.distplot(training['BalanceIncome'])
# plt.show()
training['BalanceIncome'] = np.log(training['BalanceIncome'])
test['BalanceIncome'] = np.log(test['BalanceIncome'])
plt.figure(6)
sns.distplot(training['BalanceIncome'])
# plt.show()

'''#Drop the variables used to create the new features because correlation between old features and new features with is
very high because logistic regression assumes that the variables are not highly correlated. We also want to remove
# the noise from the dataset, so removing correlated features will help in reducing the noise.
#Uncomment the 2 lines below when testing Logistic Regression Model'''

# training = training.drop(['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term'], axis=1)
# test = test.drop(['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term'], axis=1)

#Model Building PART II
print('\n Model Building PART II\n')
print('Logistic Regression:')
#Logistic Regression
X = training.drop(['Loan_Status'], axis=1)
y = training.Loan_Status

#Logistic Regression
i=1
for train_index, test_index in kf.split(X, y):
    print(f'\n{i} of kfold {kf.n_splits}')
    xtr, xvl = X.iloc[train_index], X.iloc[test_index]
    ytr, yvl = y.iloc[train_index], y.iloc[test_index]

    model = LogisticRegression(random_state=1)
    model.fit(xtr, ytr)
    pred_test = model.predict(xvl)
    score = accuracy_score(yvl, pred_test)
    print('accuracy score', score)
    i += 1
    pred_test = model.predict(test)

'''#Decision Tree
# Supervised learning algorithm used in classification problems.
# Split the sample in 2 or more homogenous sets based on the most significant splitter/ differentiator in input variables
# Uses multiple algorithms to decide to split a node in two ro more sub-nodes.
# More sub-nodes = increase in homogeneity of resultant sub-nodes (Purity increases with respect to target variable)'''
from sklearn import tree
print('\nDecision Tree:')

i=1
for train_index, test_index in kf.split(X, y):
    print(f'\n{i} of kfold {kf.n_splits}')
    xtr, xvl = X.iloc[train_index], X.iloc[test_index]
    ytr, yvl = y.iloc[train_index], y.iloc[test_index]

    model = tree.DecisionTreeClassifier(random_state=1)
    model.fit(xtr, ytr)
    pred_test = model.predict(xvl)
    score = accuracy_score(yvl, pred_test)
    print('accuracy score', score)
    pred = model.predict(test)
    i += 1

'''#Random Forest
# Tree-based bootstrapping algorithm wherein a certain no. of weak learners are combined to make a powerful prediction
# model.
# For every learner, a random sample of rows and a few randomly chosen variables are used to build a decision
# tree model
# Final prediction can be a function of all the predictions made by the individual learners
# In the case of regression problem, the final prediction can be the mean of all the predictions'''

from sklearn.ensemble import RandomForestClassifier
print('\nRandom Forest:')

i=1
for train_index, test_index in kf.split(X, y):
    print(f'\n{i} of kfold {kf.n_splits}')
    xtr, xvl = X.iloc[train_index], X.iloc[test_index]
    ytr, yvl = y.iloc[train_index], y.iloc[test_index]

    model = RandomForestClassifier(random_state=1, max_depth=10)
    model.fit(xtr, ytr)
    pred_test = model.predict(xvl)
    score = accuracy_score(yvl, pred_test)
    print('accuracy score', score)
    i += 1
    pred_test = model.predict(test)

#Improving the model with Grid Search (Hypertuning parameters)

from sklearn.model_selection import GridSearchCV

# paramgrid = {'max_depth': list(range(1, 20, 2)), 'n_estimators': list(range(1, 200, 20))}
# grid_search = GridSearchCV(RandomForestClassifier(random_state=1), paramgrid)
#
# x_train, cv_train, y_train, y_cv = train_test_split(X, y, test_size=0.3, random_state=1)
#
# grid_search.fit(x_train, y_train)
#
# GridSearchCV(cv=None, error_score='raise',
#              estimator=RandomForestClassifier(bootstrap=True, class_weight=None,criterion='gini', max_depth=None,
#              max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_samples_leaf=1,
#              min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, oob_score=False,
#              random_state=1, verbose=0, warm_start=False),
#              n_jobs=1,
#              param_grid={'max_depth': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
#                          'n_estimators': [1, 21, 41, 61, 81, 101, 121, 141, 181]},
#              pre_dispatch='2*n_jobs', refit=True, return_train_score='warn', scoring=None, verbose=0)
#
# print(grid_search.best_estimator_, "\n")

print("\nRandom Forest with optimized parameters: \n")
i=1
for train_index, test_index in kf.split(X, y):
    print(f'\n{i} of kfold {kf.n_splits}')
    xtr, xvl = X.iloc[train_index], X.iloc[test_index]
    ytr, yvl = y.iloc[train_index], y.iloc[test_index]

    model = RandomForestClassifier(random_state=1, max_depth=3, n_estimators=81)
    model.fit(xtr, ytr)
    pred_test = model.predict(xvl)
    score = accuracy_score(yvl, pred_test)
    print('accuracy score', score)
    i += 1
    pred_test = model.predict(test)

#Feature importance
plt.figure(7)
important_features = pd.Series(model.feature_importances_, index=X.columns)
important_features.plot.barh(figsize=(12, 8))
# plt.show()

'''XGBOOST
Boosting Algorithm
Works only with numeric variables

Parameters for the model:
n_estimator: Number of trees for the model
max_depth'''

from xgboost import XGBClassifier
print('\n XGBOOST Model:')
i=1
for train_index, test_index in kf.split(X, y):
    print(f'\n{i} of kfold {kf.n_splits}')
    xtr, xvl = X.iloc[train_index], X.iloc[test_index]
    ytr, yvl = y.iloc[train_index], y.iloc[test_index]

    model = XGBClassifier(n_estimator=50, max_depth=4)
    model.fit(xtr, ytr)
    pred_test = model.predict(xvl)
    score = accuracy_score(yvl, pred_test)
    print('accuracy score', score)
    i += 1
    pred_test = model.predict(test)

'''Additional Considerations:
- Use Grid Search to optimize XGBOOST model
- Coming up with a better EMI formula that includes interests
- Ensemble modeling
'''