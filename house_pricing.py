# %%
import numpy as np
from sklearn.datasets import fetch_california_housing

# %%
housing = fetch_california_housing(as_frame=True)
df = housing.frame
df.head()

# %%
X = df[ 
       [ 'MedInc', 'HouseAge',
         'AveRooms', 'AveBedrms',
         'Population', 'AveOccup',	
         'Latitude', 'Longitude'
       ]
        ]
ones = np.ones((X.shape[0],1))
X = np.c_[ones, X]
y = df['MedHouseVal']

# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# Train once
theta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

# Predict using the already-computed theta
def predict(X_new, theta):
    return X_new @ theta


# %%
y_pred = predict(X_test, theta)
y_pred

# %%
RMSE = np.sqrt(np.mean((y_test - y_pred)**2))
print(f"RMSE = {RMSE:3.3}")

# %%
ss_res = np.sum((y_test - y_pred)**2)
ss_tot = np.sum((y_test - np.mean(y_test))**2)

R2 = 1 - (ss_res / ss_tot)
print(R2)


