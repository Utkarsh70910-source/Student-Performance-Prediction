from src.components.Data_ingestion import DataIngestion
from src.components.Data_transformation import DataTransformation
from src.components.Model_trainer import ModelTrainer


if __name__ == "__main__":

    print(" Training Pipeline Started...\n")

    # STEP 1: DATA INGESTION
    data_ingestion = DataIngestion()
    train_data, test_data = data_ingestion.initiate_data_ingestion()
    print(" ---Data Ingestion Completed---")

    # STEP 2: DATA TRANSFORMATION
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initialize_data_transformation(
        train_data, test_data
    )
    print("---Data Transformation Completed---")

    # STEP 3: MODEL TRAINING
    model_trainer = ModelTrainer()
    score = model_trainer.initate_model_training(train_arr, test_arr)

    print("\n-- TRAINING COMPLETED SUCCESSFULLY --")
    print(f"-----Model Score: {score}----")