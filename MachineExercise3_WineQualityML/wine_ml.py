from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# fetch dataset 
wine_quality = fetch_ucirepo(id=186) 
  
# data (as pandas dataframes) 
X = wine_quality.data.features 
y = wine_quality.data.targets


X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.3, 
    random_state=42
)

scaler = StandardScaler().set_output(transform='pandas')
X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

linearregression = LinearRegression()
linearregression.fit(X_train_scaled, y_train)

linearregression.score(X_test_scaled, y_test)

(linearregression.predict(X_test_scaled).round() - y_test).value_counts()

y_test.shape

sns.scatterplot(x=X.iloc[:, 2], y=y.values.reshape(-1))

sns.heatmap()

plt.figure(figsize=(10, 10))
sns.heatmap(pd.concat([X, y], axis=1).corr(), annot=True);


linearregression.score(X_train_scaled, y_train)         