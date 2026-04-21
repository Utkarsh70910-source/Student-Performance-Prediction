import os
import sys
import pandas as pd
import pickle

from src.exception import CustomException
from src.logger import logging


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("Artifacts", "model.pkl")
            preprocessor_path = os.path.join("Artifacts", "preprocessor.pkl")

            # Load objects
            model = pickle.load(open(model_path, "rb"))
            preprocessor = pickle.load(open(preprocessor_path, "rb"))

            #  TRANSFORM INPUT (MOST IMPORTANT)
            data_scaled = preprocessor.transform(features)

            # Predict
            preds = model.predict(data_scaled)

            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        G1,
        G2,
        studytime,
        absences,
        failures,
        sleep_hours,
        social_media,
        Medu,
        Fedu,
        famrel,
        internet,
        higher,
    ):

        self.G1 = G1
        self.G2 = G2
        self.studytime = studytime
        self.absences = absences
        self.failures = failures
        self.sleep_hours = sleep_hours
        self.social_media = social_media
        self.Medu = Medu
        self.Fedu = Fedu
        self.famrel = famrel
        self.internet = internet
        self.higher = higher

    def get_data_as_data_frame(self):
        try:
            data_dict = {
                "G1": [self.G1],
                "G2": [self.G2],
                "studytime": [self.studytime],
                "absences": [self.absences],
                "failures": [self.failures],
                "sleep_hours": [self.sleep_hours],
                "social_media": [self.social_media],
                "Medu": [self.Medu],
                "Fedu": [self.Fedu],
                "famrel": [self.famrel],
                "internet": [self.internet],
                "higher": [self.higher],
            }

            return pd.DataFrame(data_dict)

        except Exception as e:
            raise CustomException(e, sys)