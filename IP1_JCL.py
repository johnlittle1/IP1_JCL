#!/usr/bin/env python
# coding: utf-8

# # Individual Project 1
# ---
# **Author:** John Little  
# **Version:** 1.0  
# **Semester:** Fall 2024 

# In[175]:


## import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# ### Part 1

# In[176]:


plt.plot([3,7,8,9,12])
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Simple Plot')
plt.show()


# ### Part 2

# In[177]:


## create two lists of five numbers
x = [1,2,3,4,5]
y = [2,4,6,11,14]
## plot lists
plt.plot(x,y)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('X and Y Interaction')
plt.show()


# In[178]:


## create plot using plt.subplots
fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('X and Y Interaction')
plt.show()


# ### Part 3

# In[179]:


## create evenly spaced list of 100 numbers in variable 'X'
X = np.linspace(0, 100, 100)
X


# In[180]:


## create plot of X and X Squared using plt.subplots()
fig, ax = plt.subplots()
ax.plot(X, X**2)
ax.set_xlabel('X')
ax.set_ylabel('X Squared')
ax.set_title('X and X Squared Interaction')
plt.show()


# In[181]:


## create scatter plot
sns.scatterplot(x=X, y=np.exp(X))
plt.xlabel('X')
plt.ylabel('X Squared')
plt.title('X and X Squared Interaction')
plt.show()


# ### Part 4

# In[182]:


## import data
car_sales=pd.read_csv("car-sales.csv")
car_sales.head()


# In[183]:


## investigate shape of data
car_sales.shape


# In[184]:


## investigate types of data
car_sales.dtypes


# In[185]:


## make sure there's no missing data
car_sales.isnull().sum()


# In[186]:


## check the column names for future use
car_sales.columns


# ### Part 5

# In[187]:


## change formatting of 'Price' column
car_sales['Price'] = car_sales['Price'].str.replace('$', '').str.replace(',', '').astype(np.int64)
car_sales.head()


# In[188]:


## double check format change produced desired result
car_sales.dtypes


# ### Part 6

# In[189]:


## add 'Total Price' cumulative column
car_sales['Total Sales'] = car_sales['Price'].cumsum()
car_sales.head()


# ### Part 7

# In[190]:


## add 'Sale Date' column starting with today's date
car_sales['Sale Date'] = pd.date_range(start = pd.to_datetime('today'), periods = len(car_sales)).date
car_sales.head()


# ### Part 8

# In[191]:


## create matplotlib line plot of 'Price' and 'Total Sales'
plt.plot(car_sales['Price'], car_sales['Total Sales'])
plt.xlabel('Price')
plt.ylabel('Total Sales')
plt.title('Price vs Total Sales Line Plot')
plt.show()


# ### Part 9

# In[192]:


## create matplotlib scatterplot of 'Odometer' and 'Price'
plt.scatter(car_sales['Odometer (KM)'], car_sales['Price'])
plt.xlabel('Odometer (KM)')
plt.ylabel('Price')
plt.title('Odometer vs Price Scatter Plot')
plt.show()


# ### Part 10

# In[193]:


## create matplotlib bar graph of 'Make' and 'Odometer'
plt.bar(car_sales['Make'], car_sales['Odometer (KM)'])
plt.xlabel('Make')
plt.ylabel('Odometer (KM)')
plt.title('Make vs Odometer Bar Plot')
plt.show()


# ### Part 11

# In[194]:


## create histogram of 'Make'
plt.hist(car_sales['Make'])
plt.xlabel('Make')
plt.title('Histogram of Car Make')
plt.show()


# ### Part 12

# In[195]:


## define structure of subplots
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

## first plot
axes[0, 0].plot(car_sales['Price'], car_sales['Total Sales'])
axes[0, 0].set(xlabel = 'Price', ylabel = 'Total Sales', title = 'Price vs Total Sales')

## second plot
axes[0, 1].scatter(car_sales['Odometer (KM)'], car_sales['Price'])
axes[0, 1].set(xlabel = 'Odometer (KM)', ylabel = 'Price', title = 'Odometer vs Price Scatter Plot')

## third plot
axes[1, 0].bar(car_sales['Make'], car_sales['Odometer (KM)'])
axes[1, 0].set(xlabel = 'Make', ylabel = 'Odometer (KM)', title = 'Make vs Odometer Bar Plot')

## fourth plot
axes[1, 1].hist(car_sales['Make'])
axes[1, 1].set(xlabel = 'Make', title = 'Histogram of Car Make')

plt.show()


# ### Part 13

# In[196]:


## create boxplot of 'Colour' and 'Total Sales'
sns.boxplot(x=car_sales['Colour'], y=car_sales['Total Sales'])
plt.title('Sales by Colour')
plt.show()


# ### Part 14

# In[197]:


## create linear regression plot of 'Total Sales' and 'Odometer'
sns.lmplot(data = car_sales, x = 'Total Sales', y = 'Odometer (KM)', hue = 'Make', fit_reg = True)
plt.title('Relationship between Odometer and Total Sales')
plt.xlim(0,79000)
plt.ylim(0,240000)
plt.show()


# ### Part 15

# In[198]:


## create pair plot for all possible combinations
sns.pairplot(car_sales[['Odometer (KM)', 'Doors', 'Price', 'Total Sales']])
plt.show()


# ### Part 16

# In[199]:


corr_matrix=car_sales[['Total Sales','Price', 'Odometer (KM)']].corr()
sns.heatmap(corr_matrix, annot=True)
plt.show()


# ## Comment on the correlation:
# Based on the correlation data presented in the above heat map, we can tell a few things:
# - Higher priced cars tend to have higher total sales.
# - Cars with higher odometers tend to have lower total sales. 
# - Cars with higher odometers tend to have lower prices. 
