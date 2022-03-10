#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df = pd.read_csv("50_Startups.csv")
df.head()


# In[4]:


cdf = df[['R&D Spend', 'Administration', 'Marketing Spend', 'Profit']]
cdf.head(9)


# In[6]:


plt.scatter(cdf.Administration, cdf.Profit, color='blue')
plt.xlabel("Administartion")
plt.ylabel("Profit")
plt.show()


# In[7]:


msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]


# In[11]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
train_x = np.asanyarray(train[['Administration']])
train_y = np.asanyarray(train[['Profit']])

test_x = np.asanyarray(test[['Administration']])
test_y = np.asanyarray(test[['Profit']])

poly = PolynomialFeatures(degree=2)
train_x_poly = poly.fit_transform(train_x)
train_x_poly


# In[12]:


clf = linear_model.LinearRegression()
train_y_ = clf.fit(train_x_poly, train_y)

# The Coefficients

print("Coefficients:", clf.coef_)
print("Intercepts:", clf.intercept_)


# In[17]:


plt.scatter(train.Administration, train.Profit, color='green')
XX = np.arange(0.0, 10.0, 0.1)
yy = clf.intercept_[0] + clf.coef_[0][1] * XX + clf.coef_[0][2] * np.power(XX,2)

plt.plot(XX, yy, '-r')
plt.xlabel("Administration")
plt.ylabel("Profit")


# In[19]:


from sklearn.metrics import r2_score
test_x_poly = poly.fit_transform(test_x)
test_y_ = clf.predict(test_x_poly)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y, test_y_))


# In[ ]:




