
# Import libraries
import pandas as pd

"""
Build a classification model from the expression data to predict the outcome
"""

# Data collection: 

lung1 = pd.read_csv("lung/GSE87340.csv.zip")
lung2 = pd.read_csv("lung/GSE60052.csv.zip")
lung3 = pd.read_csv("lung/GSE40419.csv.zip")
lung4 = pd.read_csv("lung/GSE37764.csv.zip")

lung_1_4 = pd.concat([lung1, lung2, lung3, lung4])


# 
print(lung_1_4.iloc[:,0:10].head())


# 
print(lung_1_4.isna().sum())

#
print(lung_1_4.isna().sum().sum())