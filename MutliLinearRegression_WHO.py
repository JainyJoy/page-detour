#!/usr/bin/env python
# coding: utf-8

# # FITTING REGRESSION MODEL FOR LIFE EXPECTANCY(WHO) DATASET

# ATTRIBUTE LIST
# 
# Country   ---Country
# YearYear  ---
# Status    ---Developed or Developing status
# Lifeexpectancy  ---Life Expectancy in age
# AdultMortality  ---Adult Mortality Rates of both sexes (probability of dying between 15 and 60 years per 1000 population)
# infantdeaths   ---Number of Infant Deaths per 1000 population
# Alcohol   ---Alcohol, recorded per capita (15+) consumption (in litres of pure alcohol)
# percentageexpenditure --Expenditure on health as a percentage of Gross Domestic Product per capita(%)
# HepatitisB   ---Hepatitis B (HepB) immunization coverage among 1-year-olds (%)
# MeaslesMeasles --- number of reported cases per 1000 population
# BMIAverage ---Body Mass Index of entire population
# under-fivedeaths  ---Number of under-five deaths per 1000 population
# Polio  ---Polio (Pol3) immunization coverage among 1-year-olds (%)
# Totalexpenditure  ---General government expenditure on health as a percentage of total government expenditure (%)
# Diphtheria  ---Diphtheria tetanus toxoid and pertussis (DTP3) immunization coverage among 1-year-olds (%)
# HIV/AIDS  ---Deaths per 1 000 live births HIV/AIDS (0-4 years)
# GDP    ---Gross Domestic Product per capita (in USD)
# Population   ---Population of the country
# thinness1-19years ---Prevalence of thinness among children and adolescents for Age 10 to 19 (% )
# thinness5-9years   ---Prevalence of thinness among children for Age 5 to 9(%)
# Incomecompositionofresources ---Human Development Index in terms of income composition of resources (index ranging from 0 to 1)
# Schooling   ----Number of years of Schooling(years)
# 

# Importing dataset and getting idea about it!
# Subsetting the dataset !

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing as pre
LED=pd.read_csv("C:\\Users\\Butterfly\\Documents\\Data\\LifeExpectancyData.csv",sep='\s*,\s*',header=0,engine='python')
df=pd.DataFrame(LED)
print(df.head())
print(df.info())
#print(pd.isnull(df).sum())
#d1=df.loc[df["Status"]=="Devoloping"]
d1=df[df['Status']=='Developing']
df1=pd.DataFrame(d1)
print(df1.head())
print(df1.info())
print(df1.describe())
print(pd.isnull(df).sum())
#d2=df.loc[df["Status"]=="Devoloped"]
#df2=pd.DataFrame(d2)
#print(df2.head())


# # Exploratory data analysis

# Checking for correlation and plotting heatmap!

# In[2]:


#df2=df1.iloc[:,3:21 ]
#print(df2.head())
plt.figure(figsize=(16, 16))
corr = df1.iloc[:,3:22].corr()
print(corr)
sns.heatmap(corr,annot=True,xticklabels=corr.columns.values,yticklabels=corr.columns.values)


# Subsetting the dataset again to get only necessaary attributes for modeling

# In[3]:


#df2=df1.loc[0:,['LifeExpectancy','Adult_Mortality','percentage_expenditure','BMI_','Polio','Diphtheria_','AIDS','GDP','thinness_1to19','thinness_5to9','Income_composition_of_resources']]
#df2=df1[["LifeExpectancy","Adult_Mortality","percentage_expenditure","BMI_","Polio","Diphtheria_","AIDS","GDP","thinness_1to19","thinness_5to9","Income_composition_of_resources"]]
#df2=df1.drop(columns=['Country','Year','Status','Total_expenditure','Population'])
df2=df1.drop(columns=['Country','Year','Status'])
print(df2.head())
print(pd.isnull(df2).sum())
#print(df2.dtypes)


# Imputing Null values in the data!

# In[4]:


from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values=np.nan,strategy='median',axis=0)
imputer=imputer.fit(df2)
df3=imputer.transform(df2.values)
df4=pd.DataFrame(df3)
df4.columns=df2.columns
df4.index=df2.index
print(df4.describe())


# Boxplot Analysis!

# In[5]:


sns.boxplot(df4["LifeExpectancy"],color="skyblue")


# In[6]:


sns.boxplot(df4["Measeles_"],color="gold")


# In[7]:


sns.boxplot(df4["Adult_Mortality"] , color="olive")


# In[8]:


sns.boxplot(df4["infant_deaths"], color="gold")


# In[9]:


sns.boxplot(df4["Alcohol"],color="teal")


# In[10]:


sns.boxplot(df4["percentage_expenditure"],color="skyblue")


# In[11]:


sns.boxplot(df4["Hepatitis_B"],color="olive")


# In[12]:


sns.boxplot(df4["Polio"],color="teal")


# In[13]:


sns.boxplot(df4["GDP"],color="teal")


# In[14]:


sns.boxplot(df4["Population"],color="skyblue")


# In[15]:


sns.boxplot(df4["Income_composition_of_resources"],color="teal")


# In[16]:


sns.boxplot(df4["BMI_"],color="gold")


# In[17]:


sns.boxplot(df4["thinness_1to19"],color="olive")


# In[18]:


sns.boxplot(df4["AIDS"],color="green")


# In[19]:


sns.boxplot(df4["UnderFiveDeaths"],color="skyblue")


# In[20]:


sns.boxplot(df4["Diphtheria_"],color="teal")


# Histogram Analysis!

# In[21]:


df4.hist(figsize=(15,15),grid=True,layout=(10,2),bins=100)
plt.show()


# Scatter Plot Analysis!

# In[22]:


g = sns.PairGrid(df4)
g.map(plt.scatter);
plt.show()


# # Modeling

# Splitting data into Test and Train set!
# Scaling the data!

# In[23]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
y=df4['LifeExpectancy']
#print(y)
x=df4.loc[:,df4.columns!='LifeExpectancy']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)
Scaler=StandardScaler()
x_train=Scaler.fit_transform(x_train)
x_test=Scaler.fit_transform(x_test)


# Fitting Regression model using linear_model()!

# In[24]:


from sklearn import linear_model
reg=linear_model.LinearRegression()
model=reg.fit(x_train,y_train)
pred=reg.predict(x_test)
print ('Coefficients: ', reg.coef_)
print ('Intercept: ',reg.intercept_)
model.score(x_train,y_train)


# Plotting Fitted Vs Actual values!

# In[25]:


plt.scatter(reg.predict(x_test),y_test)
plt.xlabel("predicted values")
plt.ylabel("actual values")
plt.show()


# Checking Goodness of the model!

# In[26]:


from sklearn.metrics import r2_score,mean_squared_error
print("R square for testing set:",r2_score(y_test,pred))
print("R square for training set:",r2_score(y_train,model.predict(x_train)))
print("Mean absolute error: %.2f" % np.mean(np.absolute(y_test - pred)))

print("Residual sum of squares (MSE): %.2f" % np.mean((y_test - pred)))


# Going for Ordinary Least Square Modeling !

# In[27]:


import statsmodels.api as st
#from statsmodels.sandbox.regression.predstd import wls_prediction_std
y=np.array(y).reshape(-1,1)
x=Scaler.fit_transform(x)
y=Scaler.fit_transform(y)
model1=st.OLS(y,x)
result=model1.fit()
print(result.summary())


# Remodeling after removing unnecessary attributes!

# In[28]:


#x=df4.drop['LifeExpectancy','Total_expenditure','GDP','Population',' thinness_1to19',' thinness_5to9']
#print(x)
x2=df4.drop(columns=['LifeExpectancy','Total_expenditure','GDP','Population','thinness_1to19','thinness_5to9'])
y2=df4['LifeExpectancy']
y2=np.array(y2).reshape(-1,1)
x2=Scaler.fit_transform(x2)
y2=Scaler.fit_transform(y2)
model2=st.OLS(y2,x2)
result2=model2.fit()
print(result2.summary())
#pred2=model2.predict()


# Plotting Fitted Vs Actual!

# In[30]:


plt.scatter(result2.fittedvalues,y2)
plt.xlabel("predicted values")
plt.ylabel("actual values")
plt.show()


# # Diagnostic Plots

# In[31]:


pred_val = result2.fittedvalues.copy()
true_val = df4['LifeExpectancy'].values.copy()
residual = true_val - pred_val


# #Plot1 --Residual Vs Fitted Plot

# In[32]:


model_norm_residuals = result2.get_influence().resid_studentized_internal
# absolute squared normalized residuals
model_norm_residuals_abs_sqrt = np.sqrt(np.abs(model_norm_residuals))
# absolute residuals
model_abs_resid = np.abs(residual)
# leverage, from statsmodels internals
model_leverage = result2.get_influence().hat_matrix_diag
# cook's distance, from statsmodels internals
model_cooks = result2.get_influence().cooks_distance[0]
dataframe=x2=df4.drop(columns=['Total_expenditure','GDP','Population','thinness_1to19','thinness_5to9'])
plot_lm_1 = plt.figure()
plot_lm_1.axes[0] = sns.residplot(pred_val, dataframe.columns[-1], data=dataframe,lowess=True,scatter_kws={'alpha': 0.5},line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

plot_lm_1.axes[0].set_title('Residuals vs Fitted')
plot_lm_1.axes[0].set_xlabel('Fitted values')
plot_lm_1.axes[0].set_ylabel('Residuals');


# #Plot2 -- Q-Q Plot (Normality Plot)

# In[33]:


import scipy as sp
fig, ax = plt.subplots(figsize=(6,2.5))
residual=(y_test - pred)
_, (__, ___, r) = sp.stats.probplot(residual, plot=ax, fit=True)
r**2


# #Plot3 -- Scale-Location Plot

# In[34]:


plot_lm_3 = plt.figure()
plt.scatter(pred_val, model_norm_residuals_abs_sqrt, alpha=0.5);
sns.regplot(pred_val, model_norm_residuals_abs_sqrt,scatter=False,ci=False,lowess=True,line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8});
plot_lm_3.axes[0].set_title('Scale-Location')
plot_lm_3.axes[0].set_xlabel('Fitted values')
plot_lm_3.axes[0].set_ylabel('$\sqrt{|Standardized Residuals|}$');

  # annotations
abs_sq_norm_resid = np.flip(np.argsort(model_norm_residuals_abs_sqrt),0)
abs_sq_norm_resid_top_3 = abs_sq_norm_resid[:3]
for i in abs_norm_resid_top_3:
    plot_lm_3.axes[0].annotate(i,xy=(pred_val[i],model_norm_residuals_abs_sqrt[i]));


# #Plot4 -- Leverage Plot

# In[35]:


plot_lm_4 = plt.figure();
plt.scatter(model_leverage, model_norm_residuals, alpha=0.5);
sns.regplot(model_leverage, model_norm_residuals,scatter=False,ci=False,lowess=True,line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8});
plot_lm_4.axes[0].set_xlim(0, max(model_leverage)+0.01)
plot_lm_4.axes[0].set_ylim(-3, 5)
plot_lm_4.axes[0].set_title('Residuals vs Leverage')
plot_lm_4.axes[0].set_xlabel('Leverage')
plot_lm_4.axes[0].set_ylabel('Standardized Residuals');

  # annotations
leverage_top_3 = np.flip(np.argsort(model_cooks), 0)[:3]
for i in leverage_top_3:
    plot_lm_4.axes[0].annotate(i,xy=(model_leverage[i],model_norm_residuals[i]));

