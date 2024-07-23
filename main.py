import argparse
import os

import pandas as pd
import torch

from DatasetLoad import DatasetLoad
from MetadataExtractor import MetadataExtractor
from utils import print_demographic_distribution


def augment_with_metadata_and_topic(dataset, extractor, gender_labels, race_labels, file_path, debug=False):
    if os.path.exists(file_path):
        print(f"Loading augmented dataset from {file_path}")
        augmented_dataset = pd.read_csv(file_path)
    else:
        print(f"Augmenting dataset and saving to {file_path}")
        total_rows = len(dataset)
        count = 0
        for index, row in dataset.iterrows():
            gender = extractor.extract_gender(row['text'])
            race = extractor.extract_race(row['text'])
            for label in gender_labels:
                dataset.at[index, label] = 1 if gender == label else 0
            for label in race_labels:
                dataset.at[index, label] = 1 if race == label else 0
            if debug:
                percentage_complete = ((count + 1) / total_rows) * 100
                print(f"Text: {row['text']}")
                print(f"Generated Metadata: Gender - {gender}, Race - {race}")
                print(f"Percentage of Completion: {percentage_complete:.2f}%, {count + 1} of {total_rows}")
            count += 1
        dataset.to_csv(file_path, index=False)
        augmented_dataset = dataset
    return augmented_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load dataset')
    parser.add_argument('--dataset_type', type=str, default='emotion', choices=['emotion', 'sarcasm'],
                        help='Type of dataset to load')
    parser.add_argument('--debug', type=bool, default=False,
                        help='Enable debug mode to print additional information')
    parser.add_argument('--percentage', type=float, default=100.0,
                        help='Percentage of the dataset to use (e.g., 0.1 for 0.1%)')

    args = parser.parse_args()
    print("Debugging is set to: ", args.debug)
    print("Percentage is set to: ", args.percentage)

    print("Torch is available: ", torch.cuda.is_available())
    print("Torch device count: ", torch.cuda.device_count())
    print("Torch available device: ", torch.cuda.get_device_name(torch.cuda.current_device()))

    # Determine the base path relative to the location of the main.py script
    base_path = os.path.dirname(os.path.abspath(__file__))

    dataset_loader = DatasetLoad(args.dataset_type, base_path, percentage=args.percentage)
    dataset_loader.load_datasets()

    extractor = MetadataExtractor()
    gender_labels = ["male", "female", "unknown"]
    race_labels = ["white", "black", "asian", "hispanic", "other", "non-identified"]

    original_train_data = dataset_loader.train_data
    original_test_data = dataset_loader.test_data
    original_val_data = dataset_loader.val_data

    train_file_path = os.path.join(base_path, f"augmented_train_{args.dataset_type}_{args.percentage}.csv")
    test_file_path = os.path.join(base_path, f"augmented_test_{args.dataset_type}_{args.percentage}.csv")
    val_file_path = os.path.join(base_path, f"augmented_val_{args.dataset_type}_{args.percentage}.csv")

    augmented_train_data = augment_with_metadata_and_topic(original_train_data.copy(), extractor, gender_labels,
                                                           race_labels, train_file_path, debug=args.debug)
    augmented_test_data = augment_with_metadata_and_topic(original_test_data.copy(), extractor, gender_labels,
                                                          race_labels, test_file_path, debug=args.debug)
    augmented_val_data = augment_with_metadata_and_topic(original_val_data.copy(), extractor, gender_labels,
                                                         race_labels, val_file_path, debug=args.debug)

    print("Train Data")
    print_demographic_distribution(augmented_train_data)

    print("\nValidation Data")
    print_demographic_distribution(augmented_val_data)

    print("\nTest Data")
    print_demographic_distribution(augmented_test_data)
