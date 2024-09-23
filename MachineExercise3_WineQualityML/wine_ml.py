from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import (train_test_split, 
                                     KFold,  
                                     GridSearchCV)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# fetch dataset 
wine_quality = fetch_ucirepo(id=186)
  
# data (as pandas dataframes) 
X = wine_quality.data.features 
yq = wine_quality.data.targets
yc = wine_quality.data.original.color

X_train, X_test, yc_train, yc_test = train_test_split(
    X,
    yc,
    test_size=0.3,
    random_state=42
)

pipeline = Pipeline([
       ('scaler', StandardScaler().set_output(transform='pandas')),
       ('model', LogisticRegression(max_iter=1000))
])

param_grid = {
    'model__C': np.logspace(-2, 2, 10),
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)

gridsearch = GridSearchCV(pipeline, param_grid, cv=kf)
gridsearch.fit(X_train, yc_train)

model = gridsearch.best_estimator_
yc_pred = model.predict(X_test)

_ = ConfusionMatrixDisplay.from_estimator(model, X_test, yc_test)