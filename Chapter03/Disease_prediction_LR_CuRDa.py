# Import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

"""
Build a classification model from the expression data to predict the outcome
"""

# Data collection: 

## Load the data and concatenate them into a single DataFrame
lung1 = pd.read_csv("lung/GSE87340.csv.zip")
lung2 = pd.read_csv("lung/GSE60052.csv.zip")
lung3 = pd.read_csv("lung/GSE40419.csv.zip")
lung4 = pd.read_csv("lung/GSE37764.csv.zip")
lung_1_4 = pd.concat([lung1, lung2, lung3, lung4])

## Print the first 5 rows and 10 columns
print(lung_1_4.iloc[:,0:10].head())

# Data Preprocessing:

## Print the total number of missing values for each columns
print(lung_1_4.isna().sum())

## Print the total number of missing values for all gene expression columns combined together
print(lung_1_4.isna().sum().sum())

# EDA

## plotting the distribution of samples corresponding to each lung cancer type
df = lung_1_4['class'].value_counts().reset_index()

## visualize the classes
sns.barplot(x = "class", y = "index", data=df)
plt.xlabel("Number of samples")
plt.ylabel("Class")
plt.show()

## Look at the different classes closely
print(set(lung_1_4['class']))

## rename those right away using the following replace method
lung_1_4['class'] = lung_1_4['class'].replace(' Normal', 'Normal')
lung_1_4['class'] = lung_1_4['class'].replace(' Tumor', 'Tumor')

## plotting the distribution of samples corresponding to each lung cancer type
df = lung_1_4['class'].value_counts().reset_index()

## visualize the classes after fixing the columns
sns.barplot(x = "class", y = "index", data=df)
plt.xlabel("Number of samples")
plt.ylabel("Class")
plt.show()

# Data transformation

## Restrict our dataset to the first 10 columns and convert the data from wide format to long format
lung_1_4_m = pd.melt(lung_1_4.iloc[:,1:12], id_vars = "class")

## Visualization for the distribution of expression across selected samples
ax = sns.boxplot(x = "variable" , y = "value", data = lung_1_4_m, hue = "class")
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.xlabel("Genes")
plt.ylabel("Expression")
plt.show()

# Data splitting

## Drop the ID and class columns in the dataset, and convert it to a NumPy ndarray
x_data = lung_1_4.drop(['class', 'ID'], axis = 1).values

## Create a NumPy ndarray for the labels from the subset data
y_data = lung_1_4['class'].values

## Convert the categorical data to numbers
classes = lung_1_4['class'].unique().tolist()
print(classes)

## Convert the classes into ordinals
func = lambda x: classes.index(x)
y_data = np.asarray([func(i) for i in y_data], dtype ="float32")

## Train test split
X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, random_state = 42, test_size=0.25, stratify = y_data)

# Model training

## Instantiating an object using the LogisticRegression function
model_lung1 = LogisticRegression()

## Fit the training data consisting of features and labels
model_lung1.fit(X_train, y_train)

# Model evaluation

## Model predictions on a single sample
pred = model_lung1.predict(X_test[12].reshape(1,-1))
print(pred)

## Model predictions on all samples from the test data
all_pred_lung= model_lung1.predict(X_test)

## Accuracy score
print(model_lung1.score(X_test, y_test))

## Confusion matrix
cm = confusion_matrix(y_test, all_pred_lung)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
display_labels = ["Normal", 'Tumor'])
disp.plot()
plt.show()

## Classification report
classification_report(y_test, all_pred_lung)

