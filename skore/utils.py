import kagglehub
import pandas as pd
import os

def download_kaggle_dataset(kaggle_path: str) -> pd.DataFrame:
    """
    Downloads the latest version of a Kaggle dataset and loads the first file into a pandas DataFrame.

    Args:
        kaggle_path (str): The Kaggle dataset identifier (e.g., 'username/dataset-name').

    Returns:
        (pd.DataFrame): A pandas DataFrame containing the data from the first CSV file found in the downloaded dataset directory.
    """
    path = kagglehub.dataset_download("algozee/teenager-menthal-healy")

    files = os.listdir(path)
    filename = os.path.join(path, files[0])

    return pd.read_csv(filename)



