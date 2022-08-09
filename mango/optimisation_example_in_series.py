"""
Usage of Mango in the California housing dataset:
https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html
This example showcases how to use mango in series
"""

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import time
from mango import Tuner

housing = fetch_california_housing()

# create a data frame from the input data
# Note: Each value of the target corresponds to the average house value in units of 100,000
features = pd.DataFrame(housing.data, columns=housing.feature_names)
target = pd.Series(housing.target, name=housing.target_names[0])


# split the data into train, validation and test set
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.2, random_state=42)


# Define a baseline: model without hyperparameter optimisation
baseline_model = ExtraTreesRegressor(max_depth=8)  # to avoid over fitting
baseline_model.fit(x_train, np.log1p(y_train))  # usage of log1p in the target to normalize its distribution
y_pred = baseline_model.predict(x_validation)
y_pred = np.exp(y_pred) - 1  # to get the real value not in log scale
rmse = np.sqrt(mean_squared_error(y_validation, y_pred))
print('rmse of baseline is:', rmse)

# Optimization of parameters with mango

# First step: define the search space of your algorithm
# usage of range instead of uniform to ensure integers
param_space = {'n_estimators': range(50, 200),
               'max_depth': range(3, 10),
               'min_samples_split': range(int(0.01*features.shape[0]), int(0.1*features.shape[0])),
               'min_samples_leaf': range(int(0.001*features.shape[0]), int(0.01*features.shape[0])),
               'max_features': ["sqrt", "log2", "auto"]
               }

# second step: define your objective function
# If you want to do cross validation, you define it inside the objective
# In this case, we use an analogous of a 1-fold cross validation
count = 0


def objective(list_parameters):
    global x_train, y_train, x_validation, y_validation, count

    np.random.seed(seed=12345)  # to make reproducible results
    count += 1
    print('count:', count)

    results = []
    for hyper_params in list_parameters:
        model = ExtraTreesRegressor(**hyper_params)
        model.fit(x_train, np.log1p(y_train))  # usage of log1p in the target to normalize its distribution
        prediction = model.predict(x_validation)
        prediction = np.exp(prediction) - 1  # to get the real value not in log scale
        error = np.sqrt(mean_squared_error(y_validation, prediction))
        results.append(error)
    return results


# third step: run the optimisation through Tuner
start_time = time.time()
tuner = Tuner(param_space, objective, dict(num_iteration=40, initial_random=10))  # Initialize Tuner
optimisation_results = tuner.minimize()
print(f'The optimisation in series takes {(time.time()-start_time)/60.} minutes.')

# Inspect the results
print('best parameters:', optimisation_results['best_params'])
print('best accuracy:', optimisation_results['best_objective'])

# run the model with the best hyper-parameters on the test set
best_model = ExtraTreesRegressor(n_jobs=-1, **optimisation_results['best_params'])
best_model.fit(x_train, np.log1p(y_train))
y_pred = np.exp(best_model.predict(x_test)) - 1  # to get the real value not in log scale
print('rmse on test:', np.sqrt(mean_squared_error(y_test, y_pred)))

