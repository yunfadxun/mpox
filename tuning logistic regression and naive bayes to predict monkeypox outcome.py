#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
warnings.filterwarnings("ignore")


# In[5]:


df = pd.read_csv("mpox.csv")
df= df.drop("Patient_ID",axis=1)
df.info()


# In[8]:


for i in df.columns[0:10]:
    print(i, df[i].unique())


# In[15]:


df=df.replace({False:0,True:1})
df["Systemic Illness"]=df["Systemic Illness"].replace({"None":0,"Fever":1,"Swollen Lymph Nodes":2,
                                                     "Muscle Aches and Pain":3})
df["MonkeyPox"]=df["MonkeyPox"].replace({"Negative":0,"Positive":1})
df


# In[21]:


for i in df.columns[0:10]:
    plt.figure()
    plt.title(f"distribution of {i}")
    sns.countplot(df[i],hue=df[i],dodge=False)


# In[23]:


#defining x and y
x=df.drop("MonkeyPox",axis=1)
y= df["MonkeyPox"].values.reshape(-1,1)


# In[24]:


#train split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split (x,y, random_state=44, train_size=0.7)


# In[25]:


#scaling
from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()
x_train= scaler.fit_transform(x_train)
x_test= scaler.fit_transform(x_test)


# In[26]:


#logreg
from sklearn.linear_model import LogisticRegression
basemodel= LogisticRegression(class_weight="balanced")
basemodel.fit(x_train, y_train)
y_base= basemodel.predict(x_test)


# In[31]:


#default model result
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
print(classification_report(y_test, y_base))
print(roc_auc_score(y_test,y_base))


# In[32]:


#defining parameters
params=({"penalty":["None","elasticnet","l1","l2"],
        "solver":["lbfgs","newton-cg","sag","saga","liblinear"],
        "tol":[10**i for i in list(range(-4,2))]+[10],
         "fit_intercept":[True,False],
         "multi_class":["auto","ovr","multinomial"],
         "verbose":[10**i for i in list(range(-4,2))]+[10],
         "warm_start":[True,False],
         "intercept_scaling":[10**i for i in list(range(-4,2))]+[10]
        })


# In[41]:


#using random search because other types (gaussian and gridsearch took longer)
from sklearn.model_selection import RandomizedSearchCV
rand_search= RandomizedSearchCV(LogisticRegression(class_weight= "balanced"), params, cv=10, n_jobs=-1, n_iter=100)
rand_search.fit(x_train,y_train)


# In[42]:


#print best params
rand_search.best_params_


# In[43]:


#use best params to predict
rand_search1 = rand_search.best_estimator_
y_predict= rand_search1.predict(x_test)


# In[44]:


#result after tuning
print(classification_report(y_test,y_predict))
print(roc_auc_score(y_test,y_predict))


# In[ ]:


#with similar result, we can't say tuning does not do much better than previous model
#let's see how it looks like using gaussian naive bayes


# In[49]:


from sklearn.naive_bayes import GaussianNB
basemodel_nb= GaussianNB()
basemodel_nb.fit(x_train,y_train)
y_base_nb = basemodel_nb.predict(x_test)
print(classification_report(y_test, y_base_nb))


# In[54]:


params_nb=({"var_smoothing": np.logspace(0,-9,num=100)})
nb_tuned= RandomizedSearchCV(GaussianNB(), params_nb, cv=5, n_iter=100, n_jobs=-1)
nb_tuned.fit(x_train,y_train)


# In[55]:


#defining best params
nb_tuned.best_params_


# In[57]:


#model with best_params
model_nb= nb_tuned.best_estimator_
y_model_nb= model_nb.predict(x_test)
print(classification_report(y_test,y_model_nb))


# In[ ]:




