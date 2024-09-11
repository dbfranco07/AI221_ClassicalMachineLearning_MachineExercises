from ucimlrepo import fetch_ucirepo

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler

wine_quality = fetch_ucirepo(id=186)

X = wine_quality.data.original

# checking the column names of the data. This includes possible features and
# target for machine learning
X.columns

# checking the shape of the data
X.shape

# Checking if there are any missing data
X.info()

# Quick checking of statistics of data
# Adding parameter include='all' to also see statistics for color -- which is a 
# categorical data
X.describe(include='all')

X['quality'].value_counts(normalize=True).plot(kind='bar')

help(pd.Series.plot)

# From the dataset, it seems that quality and color are an interesting target
# for classification based on the other features

sns.pairplot(X, hue='color')
sns.pairplot(X, hue='quality')

plt.figure(figsize=(15, 15))
sns.heatmap(X.corr(numeric_only=True), annot=True)