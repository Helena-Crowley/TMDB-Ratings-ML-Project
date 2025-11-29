from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np

# Read the data
x_train = pd.read_csv("data/X_train.csv")
x_test  = pd.read_csv("data/X_test.csv")
y_train = pd.read_csv("data/y_train.csv")
y_test  = pd.read_csv("data/y_test.csv")

# Convert y vectors
y_train = y_train.values.ravel()
y_test  = y_test.values.ravel()

# handle NaN values by replacing them with column means
imputer = SimpleImputer(strategy="mean")
x_train = imputer.fit_transform(x_train)
x_test = imputer.transform(x_test)

# ------------------------------------------------------------
# Run KNN for different k values
# ------------------------------------------------------------

best_k = None
best_rmse = float("inf")

print("KNN Regression Results:")

#k values 1-50
for k in range(1,51):
    knn = KNeighborsRegressor(
        n_neighbors=k,
        metric="euclidean",
        weights="distance"
    )

    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"k = {k:2d} -> RMSE = {rmse:.4f}")

    if rmse < best_rmse:
        best_rmse = rmse
        best_k = k

print("\nBest k:", best_k)
print(f"Best Root Mean Squared Error: {best_rmse:.4f}")
