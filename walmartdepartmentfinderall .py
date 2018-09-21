# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 08:54:07 2017

@author: Group1
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 21:24:18 2017

@author: Group1
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 02:29:18 2017

@author: Group1
"""
# Finding top 10 and bottom 10 departments for store 1
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
import random
import numpy as np
import random
from sklearn import preprocessing
from sklearn import utils
from numpy import newaxis

walmart = pd.read_csv('walmart data clean.csv', index_col = 'Store')

walmart["IsHoliday"] = walmart["IsHoliday"].astype('category')
walmart["IsHoliday"] = walmart["IsHoliday"].cat.codes
num_var = {"temp_class": {"Cold":1,"Comfortable":2,"Hot":3},
           "fuel_class": {"low":1,"Medium":2,"High":3},
           "unemploy_class": {"low":1,"Medium":2,"High":3},
           "cpi_class": {"low":1,"Medium":2,"High":3},
           "size_class":{"low":1,"Medium":2,"High":3},
           "Type":{"A":1,"B":2},
           "sales_class":{"Low":1,"Medium":2,"High":3,"Negative":0}}

walmart.replace(num_var, inplace=True)


# Make date in dataframe proper
walmart['Date'] = pd.to_datetime(walmart['Date'], format="%m/%d/%Y")

# Make dept val to name dict
#dict_dept = pd.DataFrame(dept_names.Dept_Name.values,index=dept_names.Dept).to_dict()
dept_change = {'Dept': {1: 'Candy and Tobacco',
  2: 'Health and Beauty',
  3: 'Stationery',
  4: 'Paper Goods',
  5: 'Media and Gaming',
  6: 'Cameras',
  7: 'Toys',
  8: 'Pets',
  9: 'Sporting Goods',
  10: 'Automotive',
  11: 'Hardware',
  12: 'Paint',
  13: 'Household Chemicals',
  14: 'Kitchen and Dining',
  15: 'Clinics',
  16: 'Lawn and Garden',
  17: 'Home Decor',
  18: 'Seasonal',
  19: 'Crafts and Fabrics',
  20: 'Bath and Shower',
  21: 'Books and Magazines',
  22: 'Bedding',
  23: 'Menswear',
  24: 'Boyswear',
  25: 'Shoes',
  26: 'Infant Apparel',
  27: "Ladies' Socks",
  28: 'Hosiery',
  29: 'Sleepwear/Scrubs/Underwear',
  30: 'Bras and Shapewear',
  31: 'Accessories',
  32: 'Jewelry',
  33: 'Girlswear',
  34: 'Ladieswear',
  35: 'Plus Size and Maternity',
  36: "Ladies' Outwear and Swimwear",
  37: 'Auto Services',
  38: 'Prescription Pharmacy',
  39: 'N/A',
  40: 'OTC Pharmacy',
  41: 'College/Pro Apparel (Sub 23)',
  42: 'Motor Oil (Sub 10)',
  43: 'Toys (Sub 7)',
  44: 'Crafts (Sub 19)',
  45: 'Aidco (Sub 9)',
  46: 'Cosmetics',
  47: 'Jewelry (Sub 32)',
  48: 'Firearms (Sub 9)',
  49: 'Optical',
  50: 'Optical Service Income',
  51: 'Sporting Goods (Sub 9)',
  52: 'Crafts (Sub 19)',
  53: 'Cards, Books, and Magazines (Sub 3)',
  54: 'Jewelry (Sub 32)',
  55: 'Media and Gaming (Sub 5)',
  56: 'Horticulture/Live Plants',
  57: 'Toys (Sub 7)',
  58: 'Wireless Services',
  59: 'Cosmetics/Skincare (Sub 46)',
  60: 'Concept Stores and Stamps',
  61: 'N/A',
  62: 'N/A',
  63: 'N/A',
  64: 'N/A',
  65: 'Gas',
  66: "Sam's Club",
  67: 'Celebrations',
  68: 'N/A',
  69: 'Walmart.com',
  70: "Sam's Club",
  71: 'Furniture',
  72: 'Electronics',
  73: 'Books and Magazines (Sub 21)',
  74: 'Home Management and Luggage',
  75: "Doctor's Fees",
  76: 'Academy (non-retail)',
  77: 'Large Appliances (defunct)',
  78: 'Ladieswear (Sub 34)',
  79: 'Infant Consumables and Hardlines',
  80: 'Service Deli',
  81: 'Commercial Bread',
  82: 'Impulse Merchandise and Batteries',
  83: 'Seafood (defunct)',
  84: 'Flowers and Balloons (defunct)',
  85: 'Photo Lab',
  86: 'Financial Services',
  87: 'Wireless',
  88: 'PMDC Signage (non-retail)',
  89: 'Travel Center',
  90: 'Dairy',
  91: 'Frozen Food',
  92: 'Dry Grocery',
  93: 'Fresh/Frozen Meat and Seafood',
  94: 'Produce',
  95: 'DSD Grocery, Snacks, and Beverages',
  96: 'Liquor',
  97: 'Packaged Deli',
  98: 'Bakery',
  99: 'Store Supplies (non-retail)'}}

print("Welcome to SALE$TER!")
store_num = input("Enter Store ID: ")
month = input("Enter Month number for to predict sales for: ")

# Example without predicted values
# select only store 1 data
store = walmart.loc[store_num:store_num]
# sort by weekly sales $ value
store_sorted = store.sort_values('Weekly_Sales')
# select predictor columns for RF
features = store_sorted.columns[np.r_[1,2,11:16]]

# boolean mask for each month from years provided
if month == "2":
    mask = (store_sorted['Date'] >= '2010-2-1') & (store_sorted['Date'] <= '2012-2-28')
elif month == "3":
    mask = (store_sorted['Date'] >= '2010-3-1') & (store_sorted['Date'] <= '2012-3-31')
elif month == "1":
    mask = (store_sorted['Date'] >= '2010-1-1') & (store_sorted['Date'] <= '2012-1-31')
elif month == "4":
    mask = (store_sorted['Date'] >= '2010-4-1') & (store_sorted['Date'] <= '2012-4-30')
elif month == "5":
    mask = (store_sorted['Date'] >= '2010-3-1') & (store_sorted['Date'] <= '2012-5-31')
elif month == "6":
    mask = (store_sorted['Date'] >= '2010-6-1') & (store_sorted['Date'] <= '2012-6-30')
elif month == "7":
    mask = (store_sorted['Date'] >= '2010-7-1') & (store_sorted['Date'] <= '2012-7-31')
elif month == "8":
    mask = (store_sorted['Date'] >= '2010-8-1') & (store_sorted['Date'] <= '2012-8-31')
elif month == "9":
    mask = (store_sorted['Date'] >= '2010-9-1') & (store_sorted['Date'] <= '2012-9-30')
elif month == "10":
    mask = (store_sorted['Date'] >= '2010-10-1') & (store_sorted['Date'] <= '2012-10-31')
elif month == "11":
    mask = (store_sorted['Date'] >= '2010-11-1') & (store_sorted['Date'] <= '2012-11-30')
elif month == "12":
    mask = (store_sorted['Date'] >= '2010-12-1') & (store_sorted['Date'] <= '2012-12-31')
# creating new and Final table with mask restrictions
store_comp = store_sorted.loc[mask]


#print("The bottom 10 sales departments are:\n",store_1_comp['Dept'][:10])
#print("The top 10 sales departments are:\n",store_1_comp['Dept'][-10:])

 
# Random Forest Procedure
X_train, X_test, y_train, y_test = train_test_split(store_comp[features], store_comp['sales_class'], test_size=0.4, random_state=0)

clf = RandomForestClassifier(n_jobs=2, random_state=0)
clf.fit(X_train,y_train)

RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                       max_depth=None, max_features='auto',max_leaf_nodes=None,
                       min_impurity_split=1e-07, min_samples_leaf=1,
                       min_samples_split=2, min_weight_fraction_leaf=0.0,
                       n_estimators=10, n_jobs=2, oob_score=True, random_state=0,
                       verbose=0, warm_start=False)

# Predicted sales val
pred_num = clf.predict(X_test)
# Actual sales val
#pred_num_actual = y_test
# Departments in the test group, should align with above results
test_dept = X_test['Dept']

# table with all three variables
pred_store = pd.DataFrame({'pred_sales':pred_num, 'Dept':test_dept})
# sort by sales class, ascending
pred_store_sort = pred_store.sort_values('pred_sales')

# change dept number to name
pred_store_sort.replace(dept_change, inplace=True)
# print top and bottom 10 departments1
print("The bottom 10 sales departments are:\n",pred_store_sort['Dept'][:10])
print("The top 10 sales departments are:\n",pred_store_sort['Dept'][-10:])

