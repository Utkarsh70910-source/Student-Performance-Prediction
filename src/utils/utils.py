import os
import sys
import pickle
import numpy as np

from src.logger import logging
from src.exception import CustomException

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score


# ---------------- SAVE OBJECT ---------------- #
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


# ---------------- EVALUATE MODELS ---------------- #
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        # 🔥 FORCE NUMPY (CRITICAL FIX)
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)

        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            para = param[model_name]

            logging.info(f"Training {model_name}")

            gs = GridSearchCV(model, para, cv=3)

            # ✅ SAFE FIT
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)

            model.fit(X_train, y_train)

            # ✅ SAFE PREDICT
            y_test_pred = model.predict(X_test)

            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score

        return report

    except Exception as e:
        logging.error("Exception occurred during model evaluation")
        raise CustomException(e, sys)


# ---------------- LOAD OBJECT ---------------- #
def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        logging.error("Exception occurred in load_object function")
        raise CustomException(e, sys)