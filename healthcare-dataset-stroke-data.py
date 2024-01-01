#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Important Library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


## Read CSV
df=pd.read_csv("healthcare-dataset-stroke-data.csv")


# In[3]:


## Top Five Features of the dataset
df.head()


# In[4]:


## Bottom Five Features of the dataset
df.tail()


# In[5]:


## Shape of the dataset
df.shape


# In[6]:


## Columns of the dataset
df.columns


# In[7]:


## Unique values present in each column
for i in df.columns:
    print(i,'\n')
    print(df[i].unique(),'\n')
    print('*'*50)


# In[8]:


## Value count of each unique value in each column
for i in df.columns:
    print(i,'\n')
    print(df[i].value_counts(),'\n')
    print('*'*50)


# ## Insights
# Only one datapoint have gender others so that it is not useful for any type of prediction. 

# In[9]:


## Drop Other
df.drop(index=df[df['gender']=='Other'].index,axis=0,inplace=True)
df=df.reset_index()
df.drop('index',axis=1,inplace=True)


# ## Important Observation
# Id column does not impact on the output feature so in next step we will going to remove this feature.

# In[10]:


df.drop('id',axis=1,inplace=True)


# In[11]:


## Description about the numerical features
df.describe()


# ## Insights
# From the output of the above code we got the insights about the various statistics measurement of the numerical feature.

# In[12]:


## Null value check
df.isna().sum()


# ## Insights
# From the output of the above code we got the insight that 201 datapoints are missing in bmi column.

# In[13]:


df.head()


# In[14]:


## Piecharts of Patients
fig,ax=plt.subplots(1,3,figsize=(15,20))
plt.style.use('bmh')
ax[0].set_title("Percentage of Patients")
ax[0].pie(df['gender'].value_counts(),
          autopct='%1.1f%%')
ax[0].legend(labels=df['gender'].value_counts().index)
ax[1].set_title("Percentage of Male Patients")
ax[1].pie(df[df['gender']=='Male']['stroke'].value_counts(),
        autopct='%1.1f%%')
ax[1].legend(labels=df[df['gender']=='Male']['stroke'].value_counts().index)
ax[2].set_title("Percentage of Female Patients")
ax[2].pie(df[df['gender']=='Female']['stroke'].value_counts(),
          autopct='%1.1f%%')
ax[2].legend(labels=df[df['gender']=='Female']['stroke'].value_counts().index)
plt.show()


# ## Insights
# 1. Out of total patients 58.6% are female and reset are male.
# 2. Out of total male patients 94.9% did not face any stroke but remaining 5.1% faced the stroke.
# 3. Out of total female patients 95.3% did not face any stroke but remaining 4.7% faced the stroke.

# In[15]:


## Piecharts of Patients
fig,ax=plt.subplots(1,3,figsize=(15,20))
plt.style.use("bmh")
ax[0].set_title("Percentage of Patients")
ax[0].pie(df['ever_married'].value_counts(),
          autopct='%1.1f%%')
ax[0].legend(labels=df['ever_married'].value_counts().index)
ax[1].set_title("Percentage of Married Patients")
ax[1].pie(df[df['ever_married']=='Yes']['stroke'].value_counts(),
        autopct='%1.1f%%')
ax[1].legend(labels=df[df['ever_married']=='Yes']['stroke'].value_counts().index)
ax[2].set_title("Percentage of Not married Patients")
ax[2].pie(df[df['ever_married']=='No']['stroke'].value_counts(),
          autopct='%1.1f%%')
ax[2].legend(labels=df[df['ever_married']=='No']['stroke'].value_counts().index)
plt.show()


# ## Insights
# 1. Out of total patients 65.6% are married and reset are unmarried.
# 2. Out of total married patients 93.4% did not face any stroke but remaining 6.6% faced the stroke.
# 3. Out of total unmarried patients 98.3% did not face any stroke but remaining 1.7% faced the stroke.
# 4. Married patient got more stroke than unmarried patients.

# In[16]:


## Piecharts of Patients
fig,ax=plt.subplots(1,3,figsize=(15,20))
plt.style.use("bmh")
ax[0].set_title("Percentage of Patients",fontsize=15)
ax[0].pie(df['Residence_type'].value_counts(),
          autopct='%1.1f%%')
ax[0].legend(labels=df['Residence_type'].value_counts().index)
ax[1].set_title("Percentage of Patients from Urban Area",fontsize=15)
ax[1].pie(df[df['Residence_type']=='Urban']['stroke'].value_counts(),
        autopct='%1.1f%%')
ax[1].legend(labels=df[df['Residence_type']=='Urban']['stroke'].value_counts().index)
ax[2].set_title("Percentage of Patients from Rural Area",fontsize=15)
ax[2].pie(df[df['Residence_type']=='Rural']['stroke'].value_counts(),
          autopct='%1.1f%%')
ax[2].legend(labels=df[df['Residence_type']=='Rural']['stroke'].value_counts().index)
plt.show()


# ## Insights
# 1. Out of total patients from Urban 50.8% and reset are from Rural.
# 2. Out of total patients from Urban 94.8% did not face any stroke but remaining 5.2% faced the stroke.
# 3. Out of total patients from Rural 95.5% did not face any stroke but remaining 4.5% faced the stroke.
# 4. Patients from Urban got more stroke than those of Rural.

# In[17]:


## Line chart of total patients and their work_type
y=df['work_type'].value_counts()
y1=df[df['stroke']==1]['work_type'].value_counts()
y2=round((y1/y)*100,2).fillna(0.0)
plt.style.use("bmh")
plt.figure(figsize=(6,3))
plt.title('Total patients from different work type',fontsize=12)
plt.plot(y2,marker='p',color='#E61E64')
plt.xlabel("Work Type")
plt.ylabel("Percentage")
plt.legend(['Stroke Faced','Not Stroke Faced'])
plt.show()


# ## Insights
# 1. Self-employed got maximum strokes.
# 2. Patients those never worked got minimum strokes.

# In[18]:


x=df['smoking_status'].mode()
df['smoking_status'].replace({'Unknown':x[0]},inplace=True)


# In[19]:


## Line chart of total patient and their smoking status
y=df['smoking_status'].value_counts()
y1=df[df['stroke']==1]['smoking_status'].value_counts()
y2=round((y1/y)*100,2).fillna(0.0)
plt.style.use("bmh")
plt.figure(figsize=(6,3))
plt.title('Total patients and thier smoking status',fontsize=12)
plt.plot(y2,marker='p',color='#E61E64')
plt.xlabel("Smoking status")
plt.ylabel("Percentage")
plt.legend(['Stroke Faced','Not Stroke Faced'])
plt.show()


# ## Insights
# 1. Patients those are formerly smokers got maximum strokes.
# 2. Patients those are not smoker any time got minimum strokes.

# In[20]:


## Bargraphs of gender stroke conditions
df_male=df[df['gender']=='Male'][['ever_married',
                                  'work_type',
                                  'Residence_type',
                                  'smoking_status',
                                  'stroke']].reset_index(drop=True)
df_female=df[df['gender']=='Female'][['ever_married',
                                      'work_type',
                                      'Residence_type',
                                      'smoking_status',
                                      'stroke']].reset_index(drop=True)
fig,ax=plt.subplots(2,4,figsize=(20,10))
plt.style.use("bmh")
ax[0][0].set_title("Stroke and marital status of male",fontsize=14)
ax[0][0].bar(x=df_male[df_male['stroke']==1]['ever_married'].value_counts().index,
        height=df_male[df_male['stroke']==1]['ever_married'].value_counts(),
             width=0.3,color=['#887003','#628803','#F9BB4E'])
ax[0][0].set_ylabel("Numbers")
ax[0][1].set_title("Stroke and work type of male",fontsize=14)
ax[0][1].bar(x=df_male[df_male['stroke']==1]['work_type'].value_counts().index,
        height=df_male[df_male['stroke']==1]['work_type'].value_counts(),
             width=0.3,color=['#887003','#628803','#F9BB4E'])
ax[0][2].set_title("Stroke and Residence type of male",fontsize=14)
ax[0][2].bar(x=df_male[df_male['stroke']==1]['Residence_type'].value_counts().index,
        height=df_male[df_male['stroke']==1]['Residence_type'].value_counts(),
             width=0.3,color=['#887003','#628803','#F9BB4E'])
ax[0][3].set_title("Stroke and smoking status of male",fontsize=14)
ax[0][3].bar(x=df_male[df_male['stroke']==1]['smoking_status'].value_counts().index,
        height=df_male[df_male['stroke']==1]['smoking_status'].value_counts(),
             width=0.3,color=['#887003','#628803','#F9BB4E'])
ax[1][0].set_title("Stroke and marital status of female",fontsize=14)
ax[1][0].bar(x=df_female[df_female['stroke']==1]['ever_married'].value_counts().index,
        height=df_female[df_female['stroke']==1]['ever_married'].value_counts(),
             width=0.3,color=['#887003','#628803','#F9BB4E'])
ax[1][0].set_xlabel("Marital Status")
ax[1][0].set_ylabel("Numbers")
ax[1][1].set_title("Stroke and work type of female",fontsize=14)
ax[1][1].bar(x=df_female[df_female['stroke']==1]['work_type'].value_counts().index,
        height=df_female[df_female['stroke']==1]['work_type'].value_counts(),
             width=0.3,color=['#887003','#628803','#F9BB4E'])
ax[1][1].set_xlabel("Work Type")
ax[1][2].set_title("Stroke and Residence type of female",fontsize=14)
ax[1][2].bar(x=df_female[df_female['stroke']==1]['Residence_type'].value_counts().index,
        height=df_female[df_female['stroke']==1]['Residence_type'].value_counts(),
             width=0.3,color=['#887003','#628803','#F9BB4E'])
ax[1][2].set_xlabel("Residence type")
ax[1][3].set_title("Stroke and smoking status of female",fontsize=14)
ax[1][3].bar(x=df_female[df_female['stroke']==1]['smoking_status'].value_counts().index,
        height=df_female[df_female['stroke']==1]['smoking_status'].value_counts(),
             width=0.3,color=['#887003','#628803','#F9BB4E'])
ax[1][3].set_xlabel("Residence type")
plt.show()


# In[21]:


## Independent and dependent variables
X=df.drop('stroke',axis=1)
y=df['stroke']


# In[22]:


## Train test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)


# In[23]:


num_features=X_train.dtypes[X_train.dtypes!='object'].index
cat_features=X_train.dtypes[X_train.dtypes=='object'].index


# In[24]:


## Pipeline Creation
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


# In[25]:


from sklearn.impute import SimpleImputer ## Handling missing values
from sklearn.preprocessing import OneHotEncoder ## Handling Categorical features
from sklearn.preprocessing import StandardScaler ## Feature scaling


# In[26]:


## Feature Engineering Automation
num_pipeline=Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='median')), ## missing values
    ('scaler',StandardScaler()) ##feature scaling
])
cat_pipeline=Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='most_frequent')), ## missing values
    ('encoder',OneHotEncoder()) ##Categorical to numerical
    
])                      


# In[27]:


preprocessor=ColumnTransformer([
    ('num_pipeline',num_pipeline,num_features),
    ('cat_pipeline',cat_pipeline,cat_features)
])


# In[28]:


X_train=preprocessor.fit_transform(X_train)
X_test=preprocessor.transform(X_test)


# In[29]:


X_train


# In[30]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


# In[31]:


## Automate Model Training Process
models={
    'Random Forest':RandomForestClassifier(),
    'Decision Tree':DecisionTreeClassifier(),
    'SVC':SVC()

}


# In[32]:


from sklearn.metrics import accuracy_score


# In[33]:


def evaluate_model(X_train,y_train,X_test,y_test,models):
    
    report = {}
    for i in range(len(models)):
        model = list(models.values())[i]
        # Train model
        model.fit(X_train,y_train)

            

        # Predict Testing data
        y_test_pred =model.predict(X_test)

        # Get accuracy for test data prediction
       
        test_model_score = accuracy_score(y_test,y_test_pred)

        report[list(models.keys())[i]] =  test_model_score
            

            
    return report


# In[34]:


evaluate_model(X_train,y_train,X_test,y_test,models)


# ## Final Conclusion
# 1. This dataset is used to predict whether a patient is likely to get stroke based on the input parameters like gender, age, various diseases, and smoking status.
# 2. In this ML model I got 93.81% accuracy from Random Forest Classifier.
# 3. In this ML model I got 91.00% accuracy from Decision Tree Classifier.
# 4. In this ML model I got 93.74% accuracy from Support Vector Classifier.

# In[ ]:




