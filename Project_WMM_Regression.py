#!/usr/bin/env python
# coding: utf-8

# # Project :  Dataset

# ## Load the Data
# 

# In[3]:


#load libraries
import numpy
import pandas as pd
from numpy import arange
from pandas import read_csv
from pandas import set_option
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor


# ## Load the Dataset
# 
# The dataset to be used is the Iris Dataset. This will be loaded with Pandas

# In[4]:


#load Dataset
filename_1 = 'data_1.txt'
filename_2 = 'target_1.txt'
headers_1 = ["red","blue","green","cyan","magenta","yellow","black","pink"]
headers_2 = ["x0", "y0", "Theta"]
dataset_1 = read_csv(filename_1, sep="\t", names= headers_1, index_col=False)
dataset_2 = read_csv(filename_2, sep ="\t", names = headers_2, index_col=False)
combined = pd.concat([dataset_2, dataset_1], axis=1, join_axes=[dataset_2.index])
combined["balls_visible"] = (combined.loc[:,'red':'pink']).count(axis=1)

combined.head(20)


# ## Summarizing the dataset

# ### Dimensions of the Dataset

# In[32]:


#shape
print(dataset_1.shape)


# In[12]:


#shape
print(dataset_2.shape)


# In[13]:


#head
print(dataset_1.head(5))


# In[14]:


print(dataset_2.head(5))


# ### Statistical Summary

# In[16]:


#descriptions
print(dataset_1.describe())


# In[17]:


dataset_1.count(axis=1)


# In[18]:


df_2 = dataset_2
df_2["no of balls"] = dataset_1.count(axis=1)


# In[6]:


combined_1 = combined.fillna(0)
combined_1


# In[20]:


#correlation
print(df_2.corr(method='pearson'))


# In[6]:


#class distribution
#print(dataset.groupby('class').size())


# ## Data Visualization

# ### Univariate Plot

# In[8]:


#box and whisker plots
#dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
#pyplot.show()


# In[9]:


#histograms
#dataset.hist()
#pyplot.show()


# ### Multivariate plots

# In[10]:


#scatter plot matrix
#scatter_matrix(dataset)
#pyplot.show()


# ## Evaluating some Algorithms
# 
# 1. separate out a validation dataset
# 2. Setup a test harness to use 10-fold cross -validation
# 3. Build 5 different models to predict species from flower measurements
# 4. select the best model

# ### Create a validation Dataset

# In[23]:


#split out validation dataset
array = df_2.values
X = array[:,0:3]
Y = array[:,3]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation =  train_test_split(X, Y, 
                                                                test_size = validation_size,
                                                                random_state = seed)


# In[7]:


#split out validation dataset
array = combined_1.values
X = array[:, 3:11]
Y = array[:,0:1]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)


# In[9]:


Y_validation.shape


# ### Test Harness
# 
# Using 10-fold cross validation to estimate accuracy. We will split dataset in 10 parts, train on 9 and test on 1 for all combinations of train-test splits. 
# Accuracy will be used to evaluate the models. By accuracy we mean the ratio of the number of correctly predicted instances divided by the total number of instances in the dataset multiplied by 100 resulting in a %

# In[10]:


#Test options and evaluation metric
num_folds = 10
seed = 7
scoring = 'neg_mean_squared_error'


# ### Building Models
# 

# In[11]:


# spot check Algorithms
models = []
models.append(('LR', LinearRegression()))
models.append(('LASSO', Lasso()))
models.append(('EN', ElasticNet()))
models.append(('KNN', KNeighborsRegressor()))
models.append(('CART', DecisionTreeRegressor()))
models.append(('SVR', SVR()))


# In[12]:


#Evaluate each model
results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring= scoring)
    results.append(cv_results)
    names.append(name)
    msg = "{0}: {1} ({2})". format(name, cv_results.mean(), cv_results.std())
    print(msg)


# In[13]:


#compare Algorithms
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()

