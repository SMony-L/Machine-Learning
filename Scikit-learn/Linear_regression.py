#!/usr/bin/env python3 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn import datasets, linear_model, metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# Load the data
wine = datasets.load_wine()

# Show data
print(wine['DESCR'])

# Put data into DataFrame
dfwine = pd.DataFrame(wine['data'])
dfwine.columns = wine['feature_names']
print(dfwine.head())

# Print a summary
print(dfwine.describe())

# Train test split
y = dfwine['alcohol']
X = dfwine[['malic_acid','ash', 'alcalinity_of_ash', 'magnesium',
       'total_phenols', 'flavanoids', 'nonflavanoid_phenols',
       'proanthocyanins', 'color_intensity', 'hue',
       'od280/od315_of_diluted_wines', 'proline']]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=101)

# Create and Train model
lm = LinearRegression()
lm.fit(X_train,y_train)
print('Coefficients: \n', lm.coef_)

# Practice Test Data
predictions = lm.predict(X_test)
plt.scatter(y_test, predictions)
plt.xlabel('Y Test')
plt.ylabel('Predictions')
plt.show()

# Evaluate the model
print('MSE: ', metrics.mean_squared_error(y_test, predictions))
print('RMSE: ', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

# Distribution Plot of alcohol
sns.distplot((y_test-predictions), bins=50)
plt.show()

# Coefficients
coefficients = pd.DataFrame(lm.coef_,X.columns)
coefficients.columns = ['Coefficients']
coefficients

# Pairplots
sns.pairplot(dfwine[['alcohol','malic_acid','ash', 'alcalinity_of_ash']])
plt.show()