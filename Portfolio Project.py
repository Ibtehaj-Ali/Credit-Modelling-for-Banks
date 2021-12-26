#!/usr/bin/env python
# coding: utf-8

# # Credit Risk Modelling for Banks

# This portfolio project is prepared by Ibtehaj Ali.

# ### Table of Contents
# 
# * [Abstract](#chapter1)
#     * [Credit Risk](#section_1_1)
#     * [Defining Target Class](#section_1_2)
# * [Background](#chapter2)
# * [Extracting datasets in Jupyter](#chapter5)
# * [Exploratory Analysis](#chapter6)
# * [Data Wrangling and Removing Outliers](#chapter7)
# * [Feature Engineering](#chapter8)
# * [Machine Learning Models](#chapter9)
#     * [Logistic Regression](#section_9_1)
#     * [Gradient Boosted Trees](#section_9_2)
#     * [Comparing both models](#section_9_3)

# # Abstract <a class="anchor" id="chapter1"></a>

# As an auditor, working with various clients belonging to different sectors of economy, such as financial institutions and
# manufacturing organizations, we deal with credit risk on daily basis.
# Organisation manages that risk while we analyze that amount recognized in financial statements have incorporated the effects of that credit risk. Therefore, first of all we need to understand what exactly is credit risk.

# #### Credit Risk <a class="anchor" id="section_1_1"></a>

# Credit risk is the risk of loss due to a borrower not repaying a loan. More specifically, it refers to a lender’s risk of having its cash flows interrupted when a borrower does not pay principal or interest to it.
# Let's say we lend money to an individual and we also lend money to government, by purchasing government bonds. With goernment bonds, it is highly unlikely that we would not receive our money back. While in case of an individual, the chances of not receiving our money back is high. That's mean that credit risk of individal is high while the credit risk of government is very much low.
# Taking our example forward, in case of individuals or organization with high credit risk, we have to recognized expected credit loss.

# #### Definig target class <a class="anchor" id="section_1_2"></a>
# 
# Our target class will be loan status, which indicates the credit risk pertains to each loan. The target class is defined as '0' and '1', i.e '0' being good and the '1' being bad loan status.

# In Pakistan, banks recognized expeted credit losses using guidelines given in the Prudential regulations issued by State Bank of Pakistan. Practically, banks ranks their credit risk in terms of Obligor Risk Ratings (ORR).Banks ranks each customer based on their features on he basis of the past performances of their customers.
# 
# I as a student of Big Data Analytics and data science enthusiast, will analyze credit risk on banking datasets using tools of data science and predict loan status i.e risk ratings, using Machine Learning Algorithm.

# # Background <a class="anchor" id="chapter2"></a>

# For calculating probablity of default, there are generally two types of data available. First is loan application data which contains information about customer's salary, age, background etc. Secondly, is customers repayment history.
# For this project, I will use mix of both types. This is because the both types of data alone will not be as good as combination of both.
# The dataset is available on the website of 'Datacamp'.
# 
# The dataset used in the project pertains to a financial institution, which provides credit to its customers.
# The original dataset contains more than 32,000 entries with 12 categorical /symbolic attributes. In this dataset, each entry represents a person who takes a credit by a bank, his/her age, purpose of the loan, his / income, interest rate, his default status etc.

# The attributes are:
# 
# •	age
# 
# •	income
# 
# •	home ownership status
# 
# •	employment history
# 
# •	purpose of the loan
# 
# •	loan grade
# 
# •	amount of loan
# 
# •	interest rate
# 
# •	loan status (Target Class)
# 
# •	loan to income percentage
# 
# •	default status, and
# 
# •	credit history
# 

# # Importing Libraries

# In[197]:


#Load the librarys
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
#to import plotly
import plotly.offline as py 
from plotly.offline import plot, iplot, init_notebook_mode
import plotly.graph_objs as go
init_notebook_mode(connected=True)
import plotly.tools as tls


# # Extracting datasets in Jupyter <a class="anchor" id="chapter5"></a>

# In[11]:


df=pd.read_csv('D:\BDA\Project\cr_loan2.csv')


# In[12]:


#Searching for Missings,type of data and also known the shape of data
df.info()


# In[17]:


#Looking unique values
print(df.nunique())
#Looking the data
print(df.head())


# # Exploratory Data Analysis <a class="anchor" id="chapter6"></a>

# Let's start looking at target variable i.e loan_staus and their distribution.
# I will use both plotly as well as matplotlib.

# In[106]:


Count0=df['loan_status'].value_counts()[0]
Count1=df['loan_status'].value_counts()[1]


# In[115]:


pd.Count=[Count0,Count1]


# In[118]:


labels = ['Good','Bad']


# In[123]:


colors = sns.color_palette('bright')[0:20]
#create pie chart
plt.pie(Count, labels = labels ,colors=colors, autopct='%1.1f%%',shadow=True)
plt.title('Ditribution of Target Class')
plt.show()


# The first impression of the data shows that 78.2% of the current loan status is good and 22.8% is bad.

# In[189]:


#Plotting Housing Ditribution
trace0 = go.Bar(
    x = df[df["loan_status"]== 0]["loan_intent"].value_counts().index.values,
    y = df[df["loan_status"]== 0]["loan_intent"].value_counts().values,
    name='Good'
)

trace1 = go.Bar(
    x = df[df["loan_status"]== 1]["loan_intent"].value_counts().index.values,
    y = df[df["loan_status"]== 1]["loan_intent"].value_counts().values,
    name="Bad"
)

data = [trace0, trace1]

layout = go.Layout(
    title='Loan Intention Distribuition'
)


fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='Loan Intent-Grouped')


# From the chart above, it seems that Loan intention has no direct relationship with our target class.

# In[190]:


#First plot
trace0 = go.Bar(
    x = df[df["loan_status"]== 0]["person_home_ownership"].value_counts().index.values,
    y = df[df["loan_status"]== 0]["person_home_ownership"].value_counts().values,
    name='Good'
)

#Second plot
trace1 = go.Bar(
    x = df[df["loan_status"]== 1]["person_home_ownership"].value_counts().index.values,
    y = df[df["loan_status"]== 1]["person_home_ownership"].value_counts().values,
    name="Bad"
)

data = [trace0, trace1]

layout = go.Layout(
    title='Housing Distribuition'
)


fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='Housing-Grouped')


# Form the chart above, it can be established that people who are living on rent, have bad loan status i.e high credit risk.

# In[191]:


#First plot
trace0 = go.Bar(
    x = df[df["loan_status"]== 0]["loan_grade"].value_counts().index.values,
    y = df[df["loan_status"]== 0]["loan_grade"].value_counts().values,
    name='Good'
)

#Second plot
trace1 = go.Bar(
    x = df[df["loan_status"]== 1]["loan_grade"].value_counts().index.values,
    y = df[df["loan_status"]== 1]["loan_grade"].value_counts().values,
    name="Bad"
)

data = [trace0, trace1]

layout = go.Layout(
    title='Loan Grade Distribuition'
)


fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='Loan Grade-Grouped')


# From the above graph, it can also be establised that as the loan grade falls down, the risk of default increases.

# In[198]:


df_good = df.loc[df["loan_status"] == 0]['person_age'].values.tolist()
df_bad = df.loc[df["loan_status"] == 1]['person_age'].values.tolist()
df_age = df['person_age'].values.tolist()

#First plot
trace0 = go.Histogram(
    x=df_good,
    histnorm='probability',
    name="Good Credit"
)
#Second plot
trace1 = go.Histogram(
    x=df_bad,
    histnorm='probability',
    name="Bad Credit"
)
#Third plot
trace2 = go.Histogram(
    x=df_age,
    histnorm='probability',
    name="Overall Age"
)

#Creating the grid
fig = tls.make_subplots(rows=2, cols=2, specs=[[{}, {}], [{'colspan': 2}, None]],
                          subplot_titles=('Good','Bad', 'General Distribuition'))

#setting the figs
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 2)
fig.append_trace(trace2, 2, 1)

fig['layout'].update(showlegend=True, title='Age Distribuition', bargap=0.05)
py.iplot(fig, filename='custom-sized-subplot-with-subplot-titles')


# In[126]:


# Look at the distribution of loan amounts with a histogram
n, bins, patches = plt.hist(x=df['loan_amnt'], bins='auto', color='blue' ,alpha=0.7, rwidth=0.85)
plt.xlabel("Loan Amount")
plt.title('Ditribution of Loan Amount')
plt.show()
print("Most people have loan amount between 5,000 to 10,000")


# In[20]:


# Plot a scatter plot of income against age
plt.scatter(df['person_income'], df['person_age'],c='blue', alpha=0.5)
plt.xlabel('Personal Income')
plt.ylabel('Person Age')
plt.show()
print("It can be seen that most number of people are of age group between 20 to 40. Furthermore, there are also some outliers in Age and icome columns.")


# In[127]:


import plotly.express as px
import plotly.graph_objects as go
labels = df.iloc[:,4]
values = df.iloc[:,6]
fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
fig.show()
print("The most common purpose of loan is Education and Medical ")


# In[22]:


print("Distribution of default by Purpose of loan")
print(pd.crosstab(df['loan_intent'], df['loan_status'], margins = True))


# In[23]:


print("Distribution of default by Grade of loan")
print(pd.crosstab(df['loan_grade'], df['loan_status'], margins = True))


# In[24]:


print("Distribution of default by mean of home ownership, loan status, and average percent income")
print(pd.crosstab(df['person_home_ownership'], df['loan_status'],
                  values=df['loan_percent_income'], aggfunc='mean'))


# In[25]:


print("Box plot of percentage income by loan status")
df.boxplot(column = ['loan_percent_income'], by = 'loan_status')
plt.title('Average Percent Income by Loan Status')
plt.suptitle('')
plt.show()
print("It can be seen that percentage of loan to income has direct effect on rate of default. All the deafulters of loan have the loan to income ratio of above 0.7")


# # Data Wrangling and Removing Outliers <a class="anchor" id="chapter7"></a>

# #### Removing Outliers <a class="anchor" id="section_7_1"></a>

# In[26]:


df.describe()


# In[27]:


#Numbers of outliers can be seen as maximum value in age column can not be 144, similarly employment length can not be 123.


# In[28]:


# Plot a scatter plot of employemnt length against income rates
plt.scatter(df['person_emp_length'], df['loan_int_rate'],c='blue', alpha=0.5)
plt.xlabel('Personal Employemnt Length')
plt.ylabel('Loan Interest rate')
plt.show()
print('It can be seen that there is outliers in Employment length column as the employment length can not be above 120')


# In[29]:


# Cross table for loan status, home ownership, and the max employment length
print(pd.crosstab(df['loan_status'],df['person_home_ownership'],
                  values=df['person_emp_length'], aggfunc='max'))
indices = df[df['person_emp_length'] > 60].index


# In[30]:


#Dropping the records from the data based on the indices and create a new dataframe
df_new = df.drop(indices)

# Creating the cross table from earlier and include minimum employment length
print(pd.crosstab(df_new['loan_status'],df['person_home_ownership'],
                  values=df['person_emp_length'], aggfunc='min'))


# In[31]:


# Creating the cross table from earlier and include maximum employment length
print(pd.crosstab(df_new['loan_status'],df['person_home_ownership'],
                  values=df['person_emp_length'], aggfunc='max'))


# In[32]:


# Creating the scatter plot for age and amount
plt.scatter(df['person_age'], df['loan_amnt'], c='blue', alpha=0.5)
plt.xlabel("Person Age")
plt.ylabel("Loan Amount")
plt.show()
print('As discussed above, outliers in Age coulmn can be displayed.')


# In[33]:


# droppimg the record from the data frame and create a new one
df1_new = df.drop(df[df['person_age'] > 100].index)
import matplotlib

# Creating a scatter plot of age and interest rate
colors = ["blue","red"]
plt.scatter(df1_new['person_age'], df1_new['loan_int_rate'],
            c = df1_new['loan_status'],
            cmap = matplotlib.colors.ListedColormap(colors),
            alpha=0.5)
plt.xlabel("Person Age")
plt.ylabel("Loan Interest Rate")
plt.show()


# #### Dealing with Missing Vallues <a class="anchor" id="section_7_1"></a>

# In[77]:


# Finding the number of missing values in each columns
null=df.columns[df.isnull().any()]
df[null].isnull().sum()


# In[75]:


print("895 values in person_emp_length column and 3116 values in loan_int_rate column is missing. Loan interest rate can be replaced by taking average because it haas relationship with other attributes of the datasets, but person_emp_length can be dropped because it is independent attribute.")


# In[78]:


#Replacing missing values in "loan_int_rate" Column
df['loan_int_rate'].fillna((df['loan_int_rate'].mean()), inplace =True)


# In[80]:


#Dropping missing rows in "person_emp_length" column
indices=df[df['person_emp_length'].isnull()].index
df.drop(indices, inplace = True)


# In[82]:


df.isnull().sum()


# In[83]:


# All the missing values have been removed.


# # Feature Engineering <a class="anchor" id="chapter8"></a>

# In[88]:


df.head()


# In[89]:


#person_home_ownership,loan_intent and loan_grade are non numeric columns.


# In[92]:


df['loan_intent'].head()


# In[100]:


#Separate the numeric attributes in and saving them in "df_numeric"
df_numeric=df.select_dtypes(exclude=['object'])


# In[97]:


df_numeric.head()


# In[101]:


#Separate the non-numeric attributes in and saving them in "df_non_numeric"
df_non_numeric = df.select_dtypes(include=['object'])


# In[103]:


df_non_numeric.head()


# In[102]:


#one_hot encoding the non-numeric data
df_onehot = pd.get_dummies(df_non_numeric)


# In[104]:


df_onehot.head()


# In[109]:


# Concatenating both
df_clean = pd.concat([df_numeric,df_onehot], axis=1)


# In[108]:


df_clean.head()


# # Machine Learning Models <a class="anchor" id="chapter9"></a>

# In[117]:


#Separate the target class i.e "loan_status from remaining dataset"
X=df_clean.drop('loan_status', axis=1)
Y=df_clean['loan_status']


# #### Logistic Regression <a class="anchor" id="section_9_1"></a>

# In[281]:


#Importing Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
from sklearn.metrics import roc_curve, auc, roc_auc_score


# In[120]:


# Print the parameters of the model
trainX, testX,trainY, testY = train_test_split(X,y,test_size=0.3,random_state=42)


# In[121]:


LogR = LogisticRegression(max_iter=500)


# In[124]:


LogR.fit(trainX,np.ravel(trainY))


# In[125]:


# test data prediction
predictLogR = LogR.predict(testX)


# In[ ]:


model_LR=LogisticRegression()
model_LR.fit()


# In[209]:


print(confusion_matrix(testY, predictLogR))


# In[212]:


print(accuracy_score(testY,predictLogR))
predictLogT = LogR.predict(trainX)
print(accuracy_score(trainY,predictLogT))


# In[255]:


target_names = ['Non-Default', 'Default']
print(classification_report(testY,predictLogR, target_names=target_names))


# In[229]:


# Create predictions and store them in a variable
preds = LogR.predict_proba(testX)

# Print the accuracy score the model
print(LogR.score(testX, testY))

# Plot the ROC curve of the probabilities of default
prob_default = preds[:, 1]
fallout, sensitivity, thresholds = roc_curve(testY, prob_default)
plt.plot(fallout, sensitivity, color = 'darkorange')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.show()


# #### Logistic Gradient boosted trees <a class="anchor" id="section_9_2"></a>

# In[238]:


# Importing Gradient boosted trees
from sklearn.ensemble import GradientBoostingClassifier as xgb


# In[242]:


# Train a model
Log_gbt = xgb().fit(trainX, np.ravel(trainY))

# Predict with a model
gbt_preds = Log_gbt.predict_proba(testX)


# In[249]:


# Create dataframes of first five predictions, and first five true labels
preds_df = pd.DataFrame(gbt_preds[:,1][0:5], columns = ['prob_default'])
true_df = testY.head()

# Concatenate and print the two data frames for comparison
print(pd.concat([true_df.reset_index(drop = True), preds_df], axis = 1))


# In[254]:


# Predict the labels for loan status
gbt_preds = Log_gbt.predict(testX)

# Check the values created by the predict method
print(gbt_preds)

# Print the classification report of the model
target_names = ['Non-Default', 'Default']
print(classification_report(testY, gbt_preds, target_names=target_names))


# In[262]:


prob_default2 = gbt_preds
fallout, sensitivity, thresholds = roc_curve(testY, prob_default2)
plt.plot(fallout, sensitivity, color = 'darkorange')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.show()


# #### Comparing both models <a class="anchor" id="section_9_3"></a>

# In[287]:


# Print the classification report of Logistic Regression
target_names = ['Non-Default', 'Default']
print(classification_report(testY,predictLogR, target_names=target_names))

# Print the classification report of Gradient Boosted trees
target_names = ['Non-Default', 'Default']
print(classification_report(testY, gbt_preds, target_names=target_names))


# In[275]:


# ROC chart components
fallout_lr, sensitivity_lr, thresholds_lr = roc_curve(testY, prob_default)
fallout_gbt, sensitivity_gbt, thresholds_gbt = roc_curve(testY, prob_default2)

# ROC Chart with both
plt.plot(fallout_lr, sensitivity_lr, color = 'blue', label='%s' % 'Logistic Regression')
plt.plot(fallout_gbt, sensitivity_gbt, color = 'green', label='%s' % 'GBT')
plt.plot([0, 1], [0, 1], linestyle='--', label='%s' % 'Random Prediction')
plt.title("ROC Chart for LR and GBT on the Probability of Default")
plt.xlabel('Fall-out')
plt.ylabel('Sensitivity')
plt.legend()
plt.show()


# # Conclusion <a class="anchor" id="chapter10"></a>

# The results from both the models are presented in the form of classification report. The accuracy of Logistic Regression is 81% while the accuracy of Gradient boosted tree is 93%. 
