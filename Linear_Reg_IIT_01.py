#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Linear Regression


# In[ ]:


# Write a Python program using Scikit-learn to split the attached dataset (Income & expenses data) into 80% train and 20% test data. Train or fit the data into the model and calculate the accuracy of the model using the Multiple Linear Regression Model


# In[5]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



from statsmodels.graphics.regressionplots import influence_plot
import statsmodels.formula.api as smf
import numpy as np


# In[6]:


#Read the data
df = pd.read_csv("E:\Data Science\IIT\Assign\Assign03_IIT files\Inc_Exp_Data.csv")
df.head()


# In[6]:


df.info()


# In[7]:


#check for missing values
df.isna().sum()


# In[8]:


df.corr()


# In[20]:


df.drop(['Mthly_HH_Income'], axis=1)


# In[22]:


df.drop(['Mthly_HH_Income'], axis=1).corr()


# In[28]:


#Format the plot background and scatter plots for all the variables
import seaborn as sns
sns.set(style="darkgrid", color_codes=True)
g = sns.pairplot(df, vars=["Mthly_HH_Expense", "No_of_Fly_Members","Emi_or_Rent_Amt","Annual_HH_Income"])
import matplotlib.pyplot as plt
plt.show()


# In[56]:


sns.regplot(df.Annual_HH_Income ,df.Mthly_HH_Expense, order=1, ci=None, scatter_kws={'color':'r', 's':9})
plt.xlim(0,1404000)
plt.ylim(ymin=5000);


# In[62]:


#Build model
import statsmodels.formula.api as smf 
model = smf.ols('Mthly_HH_Expense~Emi_or_Rent_Amt+No_of_Fly_Members+Annual_HH_Income',data=df).fit()


# In[63]:


model.params


# In[64]:


#t and p-Values
print(model.tvalues, '\n', model.pvalues)


# In[65]:


#R squared values
(model.rsquared,model.rsquared_adj)


# In[66]:


model.summary()


# In[68]:


print(model.summary())


# In[36]:


ml_v=smf.ols('Mthly_HH_Expense~Annual_HH_Income',data = df).fit()  
#t and p-Values
print(ml_v.tvalues, '\n', ml_v.pvalues)  


# In[38]:


ml_v.summary()


# In[37]:


ml_v1=smf.ols('Mthly_HH_Income~Annual_HH_Income',data = df).fit()  
#t and p-Values
print(ml_v1.tvalues, '\n', ml_v1.pvalues)  


# In[39]:


ml_v1.summary()


# In[74]:


# Putting feature variable to X (ie X = Dataset after removing Interest Rate)
X = df[['Emi_or_Rent_Amt','No_of_Fly_Members','Annual_HH_Income']]
# Putting response variable to y
y = df['Mthly_HH_Expense']


# In[75]:


X.head(10)


# In[76]:


#Split the Data 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8 , random_state=100)
X_train.shape,y_train.shape,X_test.shape,y_test.shape


# In[77]:


from sklearn.linear_model import LinearRegression


# In[78]:


# Representing LinearRegression as lm
lm = LinearRegression()
lm


# In[79]:


# fit the model to the training data
lm.fit(X_train,y_train)


# In[80]:


# print the intercept
print(lm.intercept_)


# In[81]:


# Let's see the coefficient
coeff_df = pd.DataFrame(lm.coef_,X_test.columns,columns=['Coefficient'])
coeff_df


# In[82]:


# Making predictions using the model
y_pred = lm.predict(X_test)
print(y_pred)


# In[83]:


#Model Performance Metrics
#Coefficient of Determination (R square)

from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)
r_squared


# In[84]:


from math import sqrt
rmse = sqrt(mse)
print('Mean_Squared_Error :' ,mse)
print('Root_Mean_Squared_Error :' ,rmse)
print('r_square_value :',r_squared)


# In[85]:


df1 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df1


# In[86]:


import statsmodels.api as sm


# In[87]:


X_train_sm = X_train # X_train is assigned to X_train_sm. Unlike SKLearn, statsmodels don't automatically fit a constant, 
#so you need to use the method sm.add_constant(X) in order to add a constant. 
X_train_sm = sm.add_constant(X_train_sm)


# In[88]:


# create a fitted model in one line
lm_1 = sm.OLS(y_train,X_train_sm).fit()


# In[89]:


# print the coefficients
lm_1.params


# In[90]:


print(lm_1.summary())


# In[40]:


import statsmodels.api as sm
qqplot=sm.qqplot(model.resid,line='q') # line = 45 to draw the diagnoal line
plt.title("Normal Q-Q plot of residuals")
plt.show()


# In[41]:


list(np.where(model.resid>5)) 


# In[42]:


model_influence = model.get_influence()
(c, _) = model_influence.cooks_distance


# In[44]:


#Plot the influencers values using stem plot
fig = plt.subplots(figsize=(20, 7))
plt.stem(np.arange(len(df)), np.round(c, 3))
plt.xlabel('Row index')
plt.ylabel('Cooks Distance')
plt.show()


# In[45]:


#index and value of influencer where c is more than .5
(np.argmax(c),np.max(c))


# In[46]:


from statsmodels.graphics.regressionplots import influence_plot
influence_plot(model)
plt.show()


# In[ ]:


#LR Model for Multiple Regression


# In[28]:



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn import metrics as sm
from sklearn import datasets


# In[7]:


#Read the data
ans2 = pd.read_csv("E:\Data Science\IIT\Assign\Assign03_IIT files\Inc_Exp_Data.csv")
ans2.head()


# In[ ]:





# In[8]:


ans2.isnull().sum()


# In[9]:


ans2.info()


# In[10]:


ans2.median() 


# In[11]:


ans2.describe()


# In[12]:


#Monthly income Versus Expenses 
sns.regplot(ans2.Mthly_HH_Income, ans2.Mthly_HH_Expense, order=1, ci=None, scatter_kws={'color':'r', 's':9}) 


# In[13]:


#Family member Versus Expenses
sns.regplot(ans2.No_of_Fly_Members, ans2.Mthly_HH_Expense, order=1, ci=None, scatter_kws={'color':'r', 's':9}) 


# In[14]:


#Rent Versus Expenses
sns.regplot(ans2.Emi_or_Rent_Amt, ans2.Mthly_HH_Expense, order=1, ci=None, scatter_kws={'color':'r', 's':9})


# In[15]:


#Rent Versus Expenses
sns.regplot( ans2.Mthly_HH_Expense,ans2.Emi_or_Rent_Amt, order=1, ci=None, scatter_kws={'color':'r', 's':9})


# In[16]:


#Rent Versus Income monthly
sns.regplot(ans2.Mthly_HH_Income,ans2.Emi_or_Rent_Amt, order=1, ci=None, scatter_kws={'color':'r', 's':9})


# In[17]:


ans2.corr()


# In[18]:



#use cmap 
sns.heatmap(ans2.corr(), cmap ='viridis') 


# In[19]:


ans2modi = ans2.drop(['Annual_HH_Income'], axis = 1,inplace=False)
ans2modi.shape


# In[20]:


sns.boxplot(ans2modi['Mthly_HH_Income']) 


# In[21]:


sns.boxplot(ans2modi['Mthly_HH_Expense']) 


# In[22]:


sns.boxplot(ans2modi['No_of_Fly_Members']) 


# In[23]:


q1 = ans2modi.quantile(.25)
q2 = ans2modi.median()
q3 = ans2modi.quantile(.75)
IQR = q3-q1
qmin = q1- 1.5*IQR 
qmax = q3+ 1.5*IQR
print("Quartile 1 = \n", q1) 
print("\n\n Quartile 2 = \n", q2) 
print("\n\n Quartile 3 = \n", q3) 
print("\n\n Inter Quartile Range = \n", IQR)
print("\n\n Q-Min = \n", qmin)
print("\n\n Q-Max = \n", qmax)


# In[24]:


Xm = ans2modi[(ans2modi.Mthly_HH_Income>qmin.Mthly_HH_Income)&(ans2modi.Mthly_HH_Income<qmax.Mthly_HH_Income)]
Xm = Xm[(Xm.Mthly_HH_Expense>qmin.Mthly_HH_Expense)&(Xm.Mthly_HH_Expense<qmax.Mthly_HH_Expense)] 
Xm = Xm[(Xm.No_of_Fly_Members>qmin.No_of_Fly_Members)&(Xm.No_of_Fly_Members<qmax.No_of_Fly_Members)]
Xm = Xm[(Xm.Emi_or_Rent_Amt>qmin.Emi_or_Rent_Amt)&(Xm.Emi_or_Rent_Amt<qmax.Emi_or_Rent_Amt)] 
print(Xm)
Xm.shape


# In[25]:


#storing training data
X_ans2 = Xm.drop(['Mthly_HH_Expense'], axis = 1,inplace=False)
X_ans2.head()


# In[26]:


#storing the target data
y_ans2= Xm['Mthly_HH_Expense']
y_ans2.head()


# In[29]:


#converting all data into standard normailization, mean = 0, sd = 1
X_ans2 = scale(X_ans2)
y_ans2 = scale(y_ans2)
print("Feature data, independent values = ",X_ans2)
print("target = ",y_ans2)


# In[30]:


Xans2_train, Xans2_test, yans2_train, yans2_test = train_test_split(X_ans2,y_ans2,test_size=0.2,random_state = 4)


# In[31]:


Xans2_train.shape, Xans2_test.shape, yans2_train.shape, yans2_test.shape


# In[32]:


# importing module
from sklearn.linear_model import LinearRegression
# creating an object of LinearRegression class
LR = LinearRegression()
# fitting the training data
LR.fit(Xans2_train,yans2_train)


# In[33]:


print("The intercept value = ",LR.intercept_)
print("Coefficients are theta0, theta1,theta2= ",LR.coef_)


# In[34]:


ans2pred = LR.predict(Xans2_test)
ans2pred


# In[35]:


from sklearn.metrics import r2_score


# In[36]:


r2_score(yans2_test,ans2pred)


# In[39]:


print("Mean squared error: %.2f" % np.mean((LR.predict(Xans2_test) - yans2_test) **2))


# In[ ]:




