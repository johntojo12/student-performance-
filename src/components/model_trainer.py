import os 
import sys
from dataclasses import dataclass

from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor


from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

# Add the parent directory of the src folder to sys.path
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self, train_array, test_array):
            
        try:
            logging.info("X Y Split started for test and train data")
            X_train, Y_train, X_test, Y_test = (
                train_array[:, :-1],  # Features for training
                train_array[:, -1],   # Target for training
                test_array[:, :-1],   # Features for testing
                test_array[:, -1]     # Target for testing
            )

            # Define the models to train
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            
            # can do same using config or yaml file 
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            # Get model evaluation results and the best model's name
            model_report, best_model_name = evaluate_models(X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test, models=models,param=params)

            # Log the performance of each model
            logging.info("Model performance metrics:")
            for model_name, metrics in model_report.items():
                logging.info(f"Model: {model_name}")
                logging.info(f"Train R²: {metrics['train_r2']:.4f}, Test R²: {metrics['test_r2']:.4f}")
                logging.info(f"Train MAE: {metrics['train_mae']:.4f}, Test MAE: {metrics['test_mae']:.4f}")
                logging.info(f"Train RMSE: {metrics['train_rmse']:.4f}, Test RMSE: {metrics['test_rmse']:.4f}")
                logging.info("-" * 50,)

            # Log the best model's selection
            logging.info(f"Best model selected: {best_model_name}")
            best_model_metrics = model_report[best_model_name]
            for metric, value in best_model_metrics.items():
                 logging.info(f"{metric}: {value}")
            logging.info("-" * 50)
            
            # Log the best parameters for the best model line by line
            logging.info(f"Best parameters for {best_model_name}:")
            best_model = models[best_model_name]
            best_params = best_model.get_params()

            # Logging each parameter in a separate line
            for param, value in best_params.items():
                logging.info(f"{param}: {value}")
            logging.info("-" * 50)

            # Save the best model to the specified file path
            best_model = models[best_model_name]
            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)

            # Make predictions with the best model
            predicted = best_model.predict(X_test)

            # Calculate R² score on the test data for the best model
            r2_square = r2_score(Y_test, predicted)
            return r2_square
        
        except Exception as e:
            raise CustomException (e,sys)
    
    
    