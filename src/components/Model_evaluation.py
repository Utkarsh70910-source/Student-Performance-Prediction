import os
import sys
from dataclasses import dataclass

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from src.utils.utils import load_object


@dataclass
class ModelEvaluationConfig:
    model_path: str = os.path.join("Artifacts", "model.pkl")
    preprocessor_path: str = os.path.join("Artifacts", "preprocessor.pkl")
    experiment_name: str = os.getenv(
        "MLFLOW_EXPERIMENT_NAME", "student-performance-prediction"
    )

class ModelEvaluation:
    def __init__(self):
        self.model_eval_config = ModelEvaluationConfig()
    
    def eval_metrics(self,actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))# here is RMSE
        mae = mean_absolute_error(actual, pred)# here is MAE
        r2 = r2_score(actual, pred)# here is r3 value
        return rmse, mae, r2
    
    def initiate_model_evaluation(self, test_array, model_path=None, preprocessor_path=None):
        try:
            model_path = model_path or self.model_eval_config.model_path
            preprocessor_path = (
                preprocessor_path or self.model_eval_config.preprocessor_path
            )

            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)

            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]
            preds = model.predict(X_test)
            rmse, mae, r2 = self.eval_metrics(y_test, preds)

            tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)

            mlflow.set_experiment(self.model_eval_config.experiment_name)
            run_name = os.getenv("MLFLOW_RUN_NAME", model.__class__.__name__)

            with mlflow.start_run(run_name=run_name):
                mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})
                mlflow.log_params(model.get_params())
                pipeline = Pipeline(
                    [("preprocessor", preprocessor), ("model", model)]
                )
                mlflow.sklearn.log_model(pipeline, artifact_path="model")

            logging.info(
                "MLflow logging completed. RMSE: %s MAE: %s R2: %s",
                rmse,
                mae,
                r2,
            )
            return rmse, mae, r2
        except Exception as e:
            logging.error("Exception occurred in model evaluation")
            raise CustomException(e, sys)
