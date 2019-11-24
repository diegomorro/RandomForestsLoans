
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


loans = pd.read_csv('loan_data.csv')


# In[6]:


loans.info()
loans.describe()


# In[10]:


loans


# In[11]:


plt.figure(figsize=(10,6))
loans[loans['credit.policy']==1]['fico'].hist(alpha=0.5,color='blue',bins=30,label='Credit.Policy=1')
loans[loans['credit.policy']==0]['fico'].hist(alpha=0.5,color='red',bins=30,label='Credit.Policy=0')
plt.legend()
plt.xlabel('FICO')


# In[23]:


plt.figure(figsize=(9,6))
loans[loans['not.fully.paid']==1]['fico'].hist(alpha=0.5,color='blue',bins=30,label='not.fully.paid=1')
loans[loans['not.fully.paid']==0]['fico'].hist(alpha=0.5,color='red',bins=30,label='not.fully.paid=0')
plt.legend()
plt.xlabel('FICO')


# In[25]:


plt.figure(figsize=(11,7))
sns.countplot(loans['purpose'], hue=loans['not.fully.paid'],palette='Set1')


# In[30]:


sns.jointplot(x='fico',y='int.rate',data=loans)


# In[27]:


plt.figure(figsize=(11,7))
sns.lmplot(y='int.rate',x='fico',data=loans,hue='credit.policy',
           col='not.fully.paid',palette='Set1')


# In[33]:


loans.info()


# In[42]:


cat_feats=['purpose']


# In[43]:


final_data = pd.get_dummies(loans,columns=cat_feats,drop_first=True)


# In[44]:


final_data.head()


# In[45]:


from sklearn.model_selection import train_test_split


# In[46]:


X_train, X_test, y_train, y_test = train_test_split(final_data.drop('not.fully.paid',axis=1), 
                                                    final_data['not.fully.paid'], test_size=0.33, random_state=42)


# In[47]:


from sklearn.tree import DecisionTreeClassifier


# In[48]:


dtree = DecisionTreeClassifier()


# In[49]:


dtree.fit(X_train, y_train)


# In[51]:


predictions = dtree.predict(X_test)


# In[52]:


from sklearn.metrics import classification_report,confusion_matrix


# In[53]:


print(classification_report(y_test, predictions))


# In[54]:


print(confusion_matrix(y_test, predictions))


# In[55]:


from sklearn.ensemble import RandomForestClassifier


# In[56]:


rfc = RandomForestClassifier(n_estimators=600)


# In[57]:


rfc.fit(X_train,y_train)


# In[60]:


predictions = rfc.predict(X_test)


# In[61]:


print(classification_report(y_test, predictions))


# In[62]:


print(confusion_matrix(y_test, predictions))

