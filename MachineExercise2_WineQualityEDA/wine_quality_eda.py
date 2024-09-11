from ucimlrepo import fetch_ucirepo
import pandas as pd

wine_quality = fetch_ucirepo(id=186)

X = wine_quality.data.features
y = wine_quality.data.targets

# checking how many different quality values are there
y.value_counts(dropna=False)

# checking the shape of the features data
X.shape

X.columns

dir(wine_quality)
wine_quality.metadata