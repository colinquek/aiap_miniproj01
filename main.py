# Standard library imports
import logging
import yaml

# Local application imports
from src.data_preparation import DataPreparation
from src.model_training import ModelTraining

RED_BOLD = "\033[1;31m"
GREEN_BOLD = "\033[1;32m"
CLEAR = "\033[0m"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def main():
    print(f"{GREEN_BOLD}[INFO] Starting End-to-End Machine Learning Pipeline...{CLEAR}")

    # Load configuration
    with open('./src/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Ingestion & Data Preparation
    data_prep = DataPreparation(config)
    raw_df = data_prep.load_raw_data()
    cleaned_df = data_prep.clean_and_engineer(raw_df)

    # Initialize ColumnTransformer Preprocessor
    X_dummy = cleaned_df.drop(columns=[config['target_column'], config['log_target_column']])
    preprocessor = data_prep.create_preprocessor(X_dummy)

    # Model Training & Hyperparameter Tuning
    trainer = ModelTraining(config, preprocessor)
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.split_data(cleaned_df)

    # Train Baseline Models
    trainer.train_baseline_models(X_train, y_train, X_val, y_val)

    # Hyperparameter Tuning
    best_model = trainer.tune_hyperparameters(X_train, y_train)

    # Final Holdout Evaluation
    print(f"{GREEN_BOLD}[INFO] Evaluating Best Tuned Model on Unseen Holdout Test Set:{CLEAR}")
    trainer.evaluate_model(best_model, X_test, y_test, label="Final Test Set")

    print(f"{GREEN_BOLD}[INFO] End-to-End ML Pipeline Execution Completed Successfully!{CLEAR}")

if __name__ == "__main__":
    main()
