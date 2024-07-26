import os

import gdown


def download_datasets():
    # URL for the new dataset
    URLS = {
        "tweets_dataset": "https://drive.google.com/file/d/1uw_FBtp7ak2B6VAyt9R7cKfiTVe2buqG/view?usp=sharing"
    }

    # Create the dataset directory if it doesn't exist
    os.makedirs("datasets", exist_ok=True)

    # Download the dataset
    for dataset_name, url in URLS.items():
        print(f"Downloading {dataset_name}")
        csv_filepath = f"datasets/{dataset_name}.csv"
        gdown.download(url, csv_filepath, fuzzy=True)
