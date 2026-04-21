import os
import sys
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("Artifacts", "raw_data.csv")
    training_data_path: str = os.path.join("Artifacts", "train_data.csv")
    test_data_path: str = os.path.join("Artifacts", "test_data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("🚀 Data ingestion started")

        try:
            # 🔥 IMPORTANT CHANGE: NEW DATASET
            data = pd.read_csv("data/stud.csv")
            logging.info(" New dataset loaded successfully")

            # Create Artifacts folder
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            # Save raw data
            data.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info(" Raw data saved")

            # Train-test split
            logging.info(" Splitting data into train and test")
            train_data, test_data = train_test_split(
                data, test_size=0.2, random_state=42
            )

            # Save train & test
            train_data.to_csv(self.ingestion_config.training_data_path, index=False)
            test_data.to_csv(self.ingestion_config.test_data_path, index=False)

            logging.info(" Train-test split completed")
            logging.info(" Data ingestion completed successfully")

            return (
                self.ingestion_config.training_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            logging.error(" Error in data ingestion")
            raise CustomException(e, sys)