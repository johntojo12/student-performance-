import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from src.logger import logging
from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, Y_train, X_test, Y_test, models,param):
    try:
        report = {}
        best_model_score = -float('inf')  # Initialize with a very low score to ensure any valid model can be better
        best_model_name = None  # Variable to hold the best model name

        for model_name, model in models.items():
            para = param.get(model_name, {})
            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,Y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,Y_train)
            
            # Predict values
            Y_train_pred = model.predict(X_train)
            Y_test_pred = model.predict(X_test)
            
            # Calculate metrics
            train_r2 = r2_score(Y_train, Y_train_pred)
            test_r2 = r2_score(Y_test, Y_test_pred)
            train_mae = mean_absolute_error(Y_train, Y_train_pred)
            test_mae = mean_absolute_error(Y_test, Y_test_pred)
            train_rmse = np.sqrt(mean_squared_error(Y_train, Y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(Y_test, Y_test_pred))
            
            # Store metrics in the dictionary
            report[model_name] = {
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse
            }

            # Condition to check if this model is the best based on the given criteria
            if (test_r2 > best_model_score and  # Test R² should be greater than the best model's test R²
                train_r2 > 0.6 and  # Train R² should be greater than 0.6
                test_mae < 10 and  # Test MAE should be less than 10
                test_rmse < 10):  # Test RMSE should be less than 10

                best_model_score = test_r2  # Update best model score with this model's test R²
                best_model_name = model_name  # Update best model name

        # If no valid model is found, raise an exception
        if best_model_name is None:
            raise CustomException("No model meets the required criteria.")

        return report, best_model_name

    except Exception as e:
        raise CustomException(e, sys)
