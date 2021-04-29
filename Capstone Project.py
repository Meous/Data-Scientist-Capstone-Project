#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import plotly as py
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff
import plotly.offline as py#visualization
py.init_notebook_mode(connected=True)#visualization
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.metrics import accuracy_score,recall_score,precision_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


# In[2]:


#read data
df = pd.read_csv('C:/Users/kwame.adu/Desktop/Kwame/Learning/Data Science/Capstone Project/bank_data.csv')


# In[3]:


#view data - first ten rows
df.head(10)


# In[4]:


#list of columns and data type
df.info()


# In[5]:


#reviewing data
df.shape


# In[6]:


df.describe()


# In[7]:


#checking cells without data
df.isnull().sum()


# In[8]:


#Descriptive statistics
#Distribution of jobs types
df_jobtype = df["job"].count()
df_jobtype


# In[9]:


#visualization of job types
jobtype = df['job'].str.split(';', expand=True).stack().value_counts().head(10)

#plotting bar graph
plt.figure(figsize=(12,12))
plt.title('Job')
jobtype.plot(kind="bar")


# In[10]:


mpl.style.use('ggplot')
churn = df[df["churn"] == 1]
active = df[df["churn"] == 0]

#function for EDA visualizations
def pie_chart(column):
    
    chart1 = go.Pie(values = churn[column].value_counts().values.tolist(),
                    labels = churn[column].value_counts().keys().tolist(),
                    name = "Churn Customers",
                    marker = dict(line = dict(width = 3,
                                             color = "rgb(243,243,243)")
                                 ),
                    hole = .4
                   )
    
    chart2 = go.Pie(values = active[column].value_counts().values.tolist(),
                    labels = active[column].value_counts().keys().tolist(),
                    name = "Active Customers",
                    marker = dict(line = dict(width = 3,
                                             color = "rgb(243,243,243)")
                                 ),
                    hole = .4
                   )
    
    data = [chart1, chart2]
    fig = go.Figure(data = data)
    #py.iplot(fig)
    
cat_cols = ["marital", "location", "job", "education" ]
num_cols = ["age", "default", "balance", "housing", "loan"]
for i in cat_cols:
    pie_chart(i)
    
for i in num_cols:
    pie_chart(i)
    
    


# In[11]:


#assigning values to churn and active customers
mpl.style.use('ggplot')
churn     = df[df["churn"] == 1]
active = df[df["churn"] == 0]
 
def plot_pie(column) :
    
    chart1 = go.Pie(values  = churn[column].value_counts().values.tolist(),
                    labels  = churn[column].value_counts().keys().tolist(),
                    domain  = dict(x = [0,.48]),
                    name    = "Churn",
                    marker  = dict(line = dict(width = 1,
                                               color = "rgb(243,243,243)")
                                  ),
                    hole    = .4
                   )
    chart2 = go.Pie(values  = active[column].value_counts().values.tolist(),
                    labels  = active[column].value_counts().keys().tolist(),
                    marker  = dict(line = dict(width = 1,
                                               color = "rgb(243,243,243)")
                                  ),
                    domain  = dict(x = [.55,1]),
                    hole    = .4,
                    name    = "Active" 
                   )
 
 
    layout = go.Layout(dict(title = column + " distribution in customer attrition ",
                            plot_bgcolor  = "rgb(243,243,243)",
                            paper_bgcolor = "rgb(243,243,243)",
                            annotations = [dict(text = "Churn",
                                                font = dict(size = 14),
                                                showarrow = False,
                                                x = .18, y = .5),
                                           dict(text = "Active",
                                                font = dict(size = 14),
                                                showarrow = False,
                                                x = .80,y = .5
                                               )
                                          ]
                           )
                      )
    data = [chart2,chart1]
    fig  = go.Figure(data = data,layout = layout)
    py.iplot(fig)
 
 
#function  for histogram for customer churn types
def histogram(column) :
    chart1 = go.Histogram(x  = churn[column],
                          histnorm= "percent",
                          name = "Churn",
                          marker = dict(line = dict(width = .4,
                                                    color = "black"
                                                    )
                                        ),
                         opacity = .8 
                         ) 
    
    chart2 = go.Histogram(x  = active[column],
                          histnorm = "percent",
                          name = "Active",
                          marker = dict(line = dict(width = .4,
                                              color = "black"
                                             )
                                 ),
                          opacity = .8
                         )
    
    data = [chart2,chart1]
    layout = go.Layout(dict(title =column + " distribution in customer attrition ",
                            plot_bgcolor  = "rgb(243,243,243)",
                            paper_bgcolor = "rgb(243,243,243)",
                            xaxis = dict(gridcolor = 'rgb(255, 255, 255)',
                                             title = column,
                                             zerolinewidth=1,
                                             ticklen=5,
                                             gridwidth=2
                                            ),
                            yaxis = dict(gridcolor = 'rgb(255, 255, 255)',
                                             title = "percent",
                                             zerolinewidth=1,
                                             ticklen=5,
                                             gridwidth=2
                                            ),
                           )
                      )
    fig  = go.Figure(data=data,layout=layout)
    
    py.iplot(fig)
    
#function  for scatter plot matrix  for numerical columns in data
def scatter_matrix(df)  :
    
    df  = df.sort_values(by = "churn" ,ascending = False)
    classes = df["churn"].unique().tolist()
    classes
    
    class_code  = {classes[k] : k for k in range(2)}
    class_code
 
    color_vals = [class_code[cl] for cl in df["churn"]]
    color_vals
 
    pl_colorscale = "Portland"
 
    pl_colorscale
 
    text = [df.loc[k,"churn"] for k in range(len(df))]
    text
 
    trace = go.Splom(dimensions = [dict(label  = "age",
                                       values = df["balance"]),
                                  dict(label  = 'balance',
                                       values = df['balance']),
                                  dict(label  = 'duration',
                                       values = df['duration'])],
                     text = text,
                     marker = dict(color = color_vals,
                                   colorscale = pl_colorscale,
                                   size = 3,
                                   showscale = False,
                                   line = dict(width = .1,
                                               color='rgb(230,230,230)'
                                              )
                                  )
                    )
    axis = dict(showline  = True,
                zeroline  = False,
                gridcolor = "#fff",
                ticklen   = 4
               )
    
    layout = go.Layout(dict(title  = 
                            "Scatter plot matrix for Numerical columns for customer attrition",
                            autosize = False,
                            height = 700,
                            width  = 700,
                            plot_bgcolor  = 'rgba(240,240,240, 0.95)',
                            xaxis1 = dict(axis),
                            yaxis1 = dict(axis),
                            xaxis2 = dict(axis),
                            yaxis2 = dict(axis),
                            xaxis3 = dict(axis),
                            yaxis3 = dict(axis),
                           )
                      )
    data   = [trace]
    fig = go.Figure(data = data,layout = layout )
    py.iplot(fig)
 
    
cat_cols = ["marital", "job", "education" ]
num_cols = ["age", "default", "balance", "housing", "loan"]
#for all categorical columns plot pie
for i in cat_cols :
    plot_pie(i)
 
#for all categorical columns plot histogram    
for i in num_cols :
    histogram(i)


# In[12]:


#correlation
corr_data = df[['products', 'duration', 'loan', 'housing', 'balance', 'default', 'age']]
correlation = corr_data.corr()
scatter_cols = correlation.columns.tolist()

corr_array = np.array(correlation)

fig = go.Figure(data=go.Heatmap(
        z = corr_array,
        x = scatter_cols,
        y = scatter_cols,
        colorscale = 'blugrn'))

fig.show()


# In[13]:


#training dataset
X = df[['age', 'default', 'balance', 'housing', 'loan']]
y = df["churn"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=42)

print("Training set has {} samples.".format(X_train.shape[0]))
print("Testing set has {} samples.".format(X_test.shape[0]))


# In[14]:


#calculating model accuracy

model = XGBClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %2f%%" % (accuracy * 100.0))


# In[15]:


#train and test data split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=56)

#model building
xgb_model = xgb.XGBClassifier(max_depth=5, learning_rate=0.08, objective= 'binary:logistic',n_jobs=-1).fit(X_train, y_train)
print('Accuracy of XGB classifier on training set: {:.2f}'
       .format(xgb_model.score(X_train, y_train)))
print('Accuracy of XGB classifier on test set: {:.2f}'
       .format(xgb_model.score(X_test[X_train.columns], y_test)))


# In[16]:


#plotting important features
fig, ax = plt.subplots(figsize=(10,8))
plot_importance(xgb_model, ax=ax)


# In[17]:


# calculating precision and recall

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

param = {'max_depth':3, 'eta':1, 'objective':'multi:softprob', 'num_class':5 }
num_round = 2
bst = xgb.train(param, dtrain, num_round)

pred = bst.predict(dtest)
improv_pred = np.asarray([np.argmax(line) for line in pred])

print("Precision = {}".format(precision_score(y_test, improv_pred, average = 'macro')))
print("Recall = {}".format(recall_score(y_test, improv_pred, average = 'macro')))


# In[18]:


#creating the churn probability
df['probability'] = xgb_model.predict_proba(df[X_train.columns])[:,1]


# In[19]:


#printing out probability column
probability = df[['customer_id', 'probability']]
probability.head(10)


# In[20]:


#export to excel
probability.to_excel("churn.xlsx")


# In[21]:


eval_set = [(X_train, y_train), (X_test, y_test)]
eval_metric = ["auc","error"]
get_ipython().run_line_magic('time', 'model.fit(X_train, y_train, eval_metric=eval_metric, eval_set=eval_set, verbose=True)')


# In[22]:


model = XGBClassifier(silent=False, 
                      scale_pos_weight=1,
                      learning_rate=0.01,  
                      colsample_bytree = 0.4,
                      subsample = 0.8,
                      objective='binary:logistic', 
                      n_estimators=1000, 
                      reg_alpha = 0.3,
                      max_depth=4, 
                      gamma=10)


# In[23]:


#Evaluation dataset
eval_set = [(X_train, y_train),(X,y)]

#defining parameters
model = xgb.XGBClassifier(subsample=1,
colsample_bytree=1,
min_child_weight=1,
max_depth=6,
learning_rate=0.3,
n_estimators=100)

#fit model
model.fit(X_train,y_train,early_stopping_rounds=10, eval_metric="error",eval_set=eval_set,verbose=0)

#making predictions
predictions = model.predict(X)
from sklearn.metrics import accuracy_score
print('Accuracy:',accuracy_score(y, predictions))


# In[28]:


#defining parameters
parameters = {"subsample":[0.5, 0.75, 1],
"colsample_bytree":[0.5, 0.75, 1],
"max_depth":[2, 6, 12],
"min_child_weight":[1,5,15],
"learning_rate":[0.3, 0.1, 0.03],
"n_estimators":[100]}


# In[29]:


#XBG model
model = xgb.XGBClassifier(n_estimators=100, n_jobs=-1)
"""Initialise Grid Search Model to inherit from the XGBoost Model,
set the of cross validations to 3 per combination and use accuracy
to score the models."""
model_gs = GridSearchCV(model,param_grid=parameters,cv=3,scoring="accuracy", use_label_encoder=False, [num_class - 1])

#Fit model
model_gs.fit(X_train,y_train,early_stopping_rounds=10, eval_metric="error",eval_set=eval_set,verbose=0)


# In[30]:


predictions = model_gs.predict(X)
print('Accuracy:',accuracy_score(y, predictions))


# In[ ]:




