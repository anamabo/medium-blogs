"""
Usage of Mango in the California housing dataset:
https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html
This example showcases how to use mango in parallel
"""

import pandas as pd
from joblib import Parallel, delayed
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import time
from mango import Tuner


class OptimizationMangoParallel:
    def __init__(self, njobs: int, configuration_params: dict, features_train, target_train,
                 features_validation, target_validation):
        self.conf_dict = configuration_params  # parameters to configure the Tuner
        self.njobs = njobs
        self.x_train = features_train
        self.y_train = target_train
        self.x_validation = features_validation
        self.y_validation = target_validation

        # Definition of the search space of the algorithm
        self.space = {'max_depth': range(3, 10),
                      'min_samples_split': range(int(0.01 * features.shape[0]), int(0.1 * features.shape[0])),
                      'min_samples_leaf': range(int(0.001 * features.shape[0]), int(0.01 * features.shape[0])),
                      'max_features': ["sqrt", "log2", "auto"]
                      }

    def _objective(self, max_depth, min_samples_split, min_samples_leaf, max_features):
        """
        This is the loss function that mango optimizes
        """
        model_parameters = {'max_depth': max_depth, 'min_samples_split': min_samples_split,
                            'min_samples_leaf': min_samples_leaf, 'max_features': max_features}
        model = ExtraTreesRegressor(**model_parameters)
        model.fit(self.x_train, np.log1p(self.y_train))  # usage of log1p in the target to normalize its distribution
        log_prediction = model.predict(self.x_validation)
        prediction = np.exp(log_prediction) - 1  # to get the real value not in log scale
        error = np.sqrt(mean_squared_error(self.y_validation, prediction))
        return error


    def _objective2(self, params_batch):
        global parameters

        results_batch = Parallel(self.njobs, backend="multiprocessing")(delayed(self._objective)(**params)
                                                                        for params in params_batch)
        rmse = [result for result in results_batch]
        return rmse

    def mango_optimization(self):
        tuner = Tuner(self.space, self._objective2, self.conf_dict)
        optimisation_results = tuner.minimize()
        return optimisation_results['best_params'], optimisation_results['best_objective']


if __name__ == '__main__':
    housing = fetch_california_housing()

    # create a data frame from the input data
    # Note: Each value of the target corresponds to the average house value in units of 100,000
    features = pd.DataFrame(housing.data, columns=housing.feature_names)
    target = pd.Series(housing.target, name=housing.target_names[0])

    # split the data into train, validation and test set
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    # Parallel optimization with Mango
    config_params = {'num_iteration': 40, 'initial_random': 10}
    start_time = time.time()
    optim = OptimizationMangoParallel(njobs=4, configuration_params=config_params,
                                      features_train=x_train, target_train=y_train,
                                      features_validation=x_validation, target_validation=y_validation)
    best_parameters, best_objective = optim.mango_optimization()
    print(f'The optimisation in parallel takes {(time.time() - start_time) / 60.} minutes.')

    # Inspect the results
    print('best parameters:', best_parameters)
    print('best accuracy:', best_objective)

    # run the model with the best hyper-parameters on the test set
    best_model = ExtraTreesRegressor(n_jobs=-1, **best_parameters)
    best_model.fit(x_train, np.log1p(y_train))
    y_pred = np.exp(best_model.predict(x_test)) - 1  # to get the real value not in log scale
    print('rmse on test:', np.sqrt(mean_squared_error(y_test, y_pred)))
