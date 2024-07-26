import os

import gdown


def download_datasets():
    # URL for the new dataset
    URLS = {
        "tweets_dataset": "https://drive.google.com/file/d/1uw_FBtp7ak2B6VAyt9R7cKfiTVe2buqG/view?usp=sharing"
    }

    # Create the dataset directory if it doesn't exist
    os.makedirs("datasets", exist_ok=True)

    # Download the dataset if it doesn't already exist
    for dataset_name, url in URLS.items():
        csv_filepath = f"datasets/{dataset_name}.csv"
        if not os.path.exists(csv_filepath):
            print(f"Downloading {dataset_name}")
            gdown.download(url, csv_filepath, fuzzy=True)
        else:
            print(f"{dataset_name} already exists. Skipping download.")