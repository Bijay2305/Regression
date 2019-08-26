Multiple Linear Regression
Now you know how to build a model with one X (feature variable) and Y (response variable). But what if you have three feature variables, or may be 10 or 100? Building a separate model for each of them, combining them, and then understanding them will be a very difficult and next to impossible task. By using multiple linear regression, you can build models between a response variable and many feature variables.
Let's see how to do that.

Step_1 : Importing and Understanding Data
In [ ]:


import pandas as pd

In [ ]:


# Importing advertising.csv
advertising_multi = pd.read_csv('advertising.csv')
# Looking at the first five rows
advertising_multi.head()



In [ ]:


# Looking at the last five rows
advertising_multi.tail()



In [ ]:


# What type of values are stored in the columns?
advertising_multi.info()



In [ ]:


# Let's look at some statistical information about our dataframe.
advertising_multi.describe()




Step_2: Visualising Data
In [ ]:


import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline



In [ ]:


# Let's plot a pair plot of all variables in our dataframe
​
#Note: Radio Vs Sales ,Newpaper Vs Sales is  scatter, Radio Vs News is scatter, 
sns.pairplot(advertising_multi)
print(advertising_multi)



In [ ]:


# Visualise the relationship between the features and the response using scatterplots
sns.pairplot(advertising_multi, x_vars=['TV','Radio','Newspaper'], y_vars='Sales',size=7, aspect=0.7, kind='scatter')




Step_3: Splitting the Data for Training and Testing
In [ ]:


# Putting feature variable to X
X = advertising_multi[['TV','Radio','Newspaper']]
​
# Putting response variable to y
y = advertising_multi['Sales']



In [ ]:


#random_state is the seed used by the random number generator. It can be any integer.
#from sklearn.cross_validation import train_test_split
# TO find oout the Y-prediction we need to test 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7 , random_state=100)
