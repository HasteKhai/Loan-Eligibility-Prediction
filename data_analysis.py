import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


#Setting up the training and test sets
training = pd.read_csv('training_set.csv')
test = pd.read_csv('test_set.csv')

train_original = training.copy()
test_original = test.copy()


'''Understanding the data'''
print(training.dtypes)
print(training.columns)
print(test.columns)

# Training set contains:
# Target variable: Loan_Status
# + 12 Independant variables

# Test set contains:
# The same features as the training set without the Loan_Status

#Number of rows and columns
print(training.shape)
print(test.shape)

'''Univariate Analysis'''

#Target Variable i.e. Loan_Status
print(training['Loan_Status'].value_counts())

#Display proportions
print(training['Loan_Status'].value_counts(normalize=True))

plt.figure(1)
training['Loan_Status'].value_counts().plot.bar()
plt.show()

#Independant Categorical Variables
plt.figure(1)
plt.subplot(221)
training['Gender'].value_counts(normalize=True).plot.bar(figsize=(20, 10), title='Gender')
plt.subplot(222)
training['Married'].value_counts(normalize=True).plot.bar(title='Married')
plt.subplot(223)
training['Self_Employed'].value_counts(normalize=True).plot.bar(title='Self Employed')
plt.subplot(224)
training['Credit_History'].value_counts(normalize=True).plot.bar(title='Credit History')
plt.show()

#Ordinal Categorical Variables
plt.figure(1)
plt.subplot(131)
training['Dependents'].value_counts(normalize=True).plot.bar(figsize=(24,6), title='Dependents')
plt.subplot(132)
training['Education'].value_counts(normalize=True).plot.bar(title='Education')
plt.subplot(133)
training['Property_Area'].value_counts(normalize=True).plot.bar(title='Property Area')
plt.show()

#Numerical Variable

#Income
plt.subplot(121)
sns.distplot(training['ApplicantIncome'])
plt.subplot(122)
training['ApplicantIncome'].plot.box(figsize=(16, 5))
plt.show()

#Segregate by Education
training.boxplot(column='ApplicantIncome', by='Education')
plt.suptitle("")
plt.show()

#Co-Applicant Income
plt.subplot(121)
sns.distplot(training['CoapplicantIncome'])
plt.subplot(122)
training['CoapplicantIncome'].plot.box(figsize=(16, 5))
plt.show()

#Loan Amount
plt.subplot(121)
sns.distplot(training['LoanAmount'])
plt.subplot(122)
training['LoanAmount'].plot.box(figsize=(16, 5))
plt.show()

'''Bivariate Analysis
- Applicants with high income should have more chance for approval
- Applicants who have repaid their previous debts should have a higher chance for approval
- Loan approval should depend on the loan amount (lower = better).
- The lower the amount paid per month, the higher chance for a loan approval should be
'''

#Relation between Categorial Variables and Loan_Status
gender = pd.crosstab(training['Gender'], training['Loan_Status'])
gender.div(gender.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4, 4))
married = pd.crosstab(training['Married'], training['Loan_Status'])
married.div(married.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4, 4))
dependents = pd.crosstab(training['Dependents'], training['Loan_Status'])
dependents.div(dependents.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4, 4))
education = pd.crosstab(training['Education'], training['Loan_Status'])
education.div(education.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4, 4))
self_employed = pd.crosstab(training['Self_Employed'], training['Loan_Status'])
self_employed.div(self_employed.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4, 4))
plt.show()

#Relation between Ordinal Categorical Variables and Loan_Status
credit_history = pd.crosstab(training['Credit_History'], training['Loan_Status'])
credit_history.div(credit_history.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
property_area = pd.crosstab(training['Property_Area'], training['Loan_Status'])
property_area.div(property_area.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.show()

#Relation between Numerical Variables and Loan_Status

#Applicant Income
training.groupby('Loan_Status')['ApplicantIncome'].mean().plot.bar()
plt.show()
#No changes in the mean income, so we apply bins
bins = [0, 2500, 4000, 6000, 81000]
training['Income_bin'] = pd.cut(training['ApplicantIncome'], bins, labels=['Low', 'Average', 'High', 'Very high'])
income_bin = pd.crosstab(training['Income_bin'], training['Loan_Status'])
income_bin.div(income_bin.sum(1).astype(float), axis=0).plot.bar(stacked=True)
plt.show()

#Co-Applicant Income
bins = [0, 1000, 3000, 42000]
training['Coapplicant_Income_bin'] = pd.cut(training['CoapplicantIncome'], bins, labels=['Low', 'Average', 'High'])
coapplicant_income_bin = pd.crosstab(training['Coapplicant_Income_bin'], training['Loan_Status'])
coapplicant_income_bin.div(coapplicant_income_bin.sum(1).astype(float), axis=0).plot.bar(stacked=True)
plt.xlabel("Co-Applicant income")
plt.show()
#Results don't really make sense (Maybe due to the fact that most of the applicants don't have a co-applicant

#Combine Applicant and Co-Applicant Incomes
training['Total_income'] = training['ApplicantIncome'] + training['CoapplicantIncome']
bins = [0, 2500, 4000, 6000, 81000]
training['Total_income_bin'] = pd.cut(training['Total_income'], bins, labels=['Low', 'Average', 'High', 'Very high'])
total_income_bin = pd.crosstab(training['Total_income_bin'], training['Loan_Status'])
total_income_bin.div(total_income_bin.sum(1).astype(float), axis=0).plot.bar(stacked=True)
plt.show()

#Loan Amount
bins = [0, 100, 200, 700]
training['Loan_amount_bin'] = pd.cut(training['LoanAmount'], bins, labels=['Low', 'Average', ' High'])
loan_amount_bin = pd.crosstab(training['Loan_amount_bin'], training['Loan_Status'])
loan_amount_bin.div(loan_amount_bin.sum(1).astype(float), axis=0).plot.bar(stacked=True)
plt.show()

#Change into numerical values and remove the bin headers
training = training.drop(['Income_bin', 'Total_income', 'Loan_amount_bin', 'Coapplicant_Income_bin',
                          'Total_income_bin'], axis=1)

training['Dependents'].replace('3+', 3, inplace=True)
test['Dependents'].replace('3+', 3, inplace=True)
training['Loan_Status'].replace('Y', 1, inplace=True)
training['Loan_Status'].replace('N', 0, inplace=True)

training2 = training.drop(['Loan_ID','Gender','Married','Education','Self_Employed','Loan_Amount_Term',
                           'Property_Area'], axis=1)


#Correlation between all the numerical variables using Heat Map
matrix = training2.corr()
f, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(matrix, vmax=.8, square=True, cmap="BuPu")
plt.show()
