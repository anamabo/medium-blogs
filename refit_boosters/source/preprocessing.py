import os
from dotenv import load_dotenv
import logging
import pandas as pd
import shutil

load_dotenv(
        os.path.join(os.getcwd(), "refit_boosters", ".env")
    )

# Import must be after the environment parameters
import kaggle

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


def check_kaggle_setup():
    kaggle_username = os.environ.get("KAGGLE_USERNAME")
    kaggle_key = os.environ.get("KAGGLE_KEY")

    if kaggle_username and kaggle_key:
        logging.info("✓ Kaggle credentials found in environment variables")
    else:
        logging.info("ℹ Environment variables not set, checking for kaggle.json file...")

    try:
        kaggle.api.authenticate()
        logging.info("✓ Kaggle API authentication successful")
        return True
    except Exception as e:
        print(f"✗ Kaggle API authentication failed: {e}")
        return False


def download_kaggle_files(data_folder: str):
    os.makedirs(data_folder, exist_ok=True)

    dataset_id = "kamilpytlak/personal-key-indicators-of-heart-disease"
    logging.info(f"Downloading dataset {dataset_id}...")
    kaggle.api.dataset_download_files(
        dataset_id,
        path=data_folder,
        unzip=True,
    )
    logging.info(f"Raw files saved in {data_folder}.")


def remove_unnecessary_data():
    shutil.rmtree(
        os.path.join(os.getcwd(), "refit_boosters", "data", "2020")
    )
    os.remove(
        os.path.join(os.getcwd(), "refit_boosters", "data", "2022", "heart_2022_with_nans.csv")
    )


def clean_data(raw_data_fname: str, output_data_folder: str, output_data_fname: str):
    logging.info(f"Reading and cleaning the file {raw_data_fname}...")
    df = pd.read_csv(raw_data_fname)

    int_cols = ['PhysicalHealthDays', 'MentalHealthDays', 'SleepHours']
    float_cols = ['HeightInMeters', 'WeightInKilograms', 'BMI']
    bool_cols = ['PhysicalActivities', 'HadAngina', 'HadStroke', 'HadAsthma', 'HadSkinCancer', 'HadCOPD',
                 'HadDepressiveDisorder', 'HadKidneyDisease', 'HadArthritis', 'DeafOrHardOfHearing',
                 'BlindOrVisionDifficulty', 'DifficultyConcentrating', 'DifficultyDressingBathing',
                 'DifficultyDressingBathing', 'DifficultyErrands', 'ChestScan', 'AlcoholDrinkers', 'HIVTesting',
                 'FluVaxLast12', 'PneumoVaxEver', 'HighRiskLastYear']
    target = ['HadHeartAttack']  # bool
    cat_cols = list(set(df.columns) - set(int_cols) - set(float_cols) - set(bool_cols) - set(target))

    df[int_cols] = df[int_cols].astype(int)
    df[float_cols] = df[float_cols].astype(float)
    df[cat_cols] = df[cat_cols].astype('category')

    # preprocess boolean
    for col in bool_cols + target:
        df[col] = (df[col] == 'Yes')

    os.makedirs(output_data_folder, exist_ok=True)
    df.to_parquet(os.path.join(output_data_folder, output_data_fname))
    logging.info(f"The cleaned file is stored in {os.path.join(output_data_folder, output_data_fname)}")


def run_preprocessing_pipeline():
    input_data_folder = os.path.join(os.getcwd(), "refit_boosters", "data")
    raw_data_filename = os.path.join(input_data_folder, "2022", "heart_2022_no_nans.csv")
    preprocessed_data_folder = os.path.join(input_data_folder, "cleaned")
    preprocessed_data_filename = "heart_2022.parquet"

    check_kaggle_setup()
    download_kaggle_files(data_folder=input_data_folder)
    remove_unnecessary_data()
    clean_data(
        raw_data_fname=raw_data_filename,
        output_data_folder=preprocessed_data_folder,
        output_data_fname=preprocessed_data_filename,
    )
    logging.info("Process Done!")


if __name__ == "__main__":
    run_preprocessing_pipeline()
