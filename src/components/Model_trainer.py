import os
import sys
import numpy as np

from dataclasses import dataclass
from sklearn.metrics import r2_score

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from src.logger import logging
from src.exception import CustomException
from src.utils.utils import evaluate_models, save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("Artifacts", "model.pkl")


class ModelTrainer:

    def __init__(self):
        self.config = ModelTrainerConfig()

    def initate_model_training(self, train_arr, test_arr):
        try:
            logging.info(" Model training started (Single Model)")

            # Split features & target
            X_train = train_arr[:, :-1]
            y_train = train_arr[:, -1]

            X_test = test_arr[:, :-1]
            y_test = test_arr[:, -1]

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Linear Regression": LinearRegression()
            }

            params = {
                "Decision Tree": {'criterion': ['squared_error']},
                "Random Forest": {'n_estimators': [32, 64]},
                "Linear Regression": {}
            }

            # Evaluate
            model_report = evaluate_models(
                X_train, y_train, X_test, y_test, models, params
            )

            print("\n Model Report:", model_report)

            # Best model
            best_score = max(model_report.values())

            best_model_name = list(model_report.keys())[ 
                list(model_report.values()).index(best_score)
            ]

            best_model = models[best_model_name]

            # Save model
            save_object(
                self.config.trained_model_file_path,
                best_model
            )

            print(f"\n Best Model: {best_model_name} ({best_score})")

            return best_score

        except Exception as e:
            raise CustomException(e, sys)