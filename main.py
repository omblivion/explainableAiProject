import argparse
import os

import pandas as pd

from DatasetLoad import DatasetLoad
from MetadataExtractor import MetadataExtractor


def augment_with_metadata_and_topic(dataset, extractor, candidate_labels, file_path, debug=False):
    if os.path.exists(file_path):
        print(f"Loading augmented dataset from {file_path}")
        augmented_dataset = pd.read_csv(file_path)
    else:
        print(f"Augmenting dataset and saving to {file_path}")
        total_rows = len(dataset)
        count = 0
        for index, row in dataset.iterrows():
            probabilities = extractor.extract_probabilities(row['text'], candidate_labels)
            for label in candidate_labels:
                dataset.at[index, label] = probabilities.get(label, 0)
            if debug:
                percentage_complete = ((count + 1) / total_rows) * 100
                print(f"Text: {row['text']}")
                print("Generated Metadata:", probabilities)
                print(f"Percentage of Completion: {percentage_complete:.2f}%, {count + 1} of {total_rows}")
            count += 1
        dataset.to_csv(file_path, index=False)
        augmented_dataset = dataset
    return augmented_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load dataset')
    parser.add_argument('--dataset_type', type=str, default='emotion', choices=['emotion', 'sarcasm'],
                        help='Type of dataset to load')
    parser.add_argument('--debug', type=bool, default=True,
                        help='Enable debug mode to print additional information')
    parser.add_argument('--percentage', type=float, default=100.0,
                        help='Percentage of the dataset to use (e.g., 0.1 for 0.1%)')

    args = parser.parse_args()
    print("Debugging is set to: ", args.debug)
    print("Percentage is set to: ", args.percentage)

    # Determine the base path relative to the location of the main.py script
    base_path = os.path.dirname(os.path.abspath(__file__))

    dataset_loader = DatasetLoad(args.dataset_type, base_path, percentage=args.percentage)
    dataset_loader.load_datasets()

    extractor = MetadataExtractor()
    candidate_labels = ["happiness", "sadness", "anger", "fear", "love", "surprise"]

    original_train_data = dataset_loader.train_data
    original_test_data = dataset_loader.test_data
    original_val_data = dataset_loader.val_data

    train_file_path = os.path.join(base_path, f"augmented_train_{args.dataset_type}_{args.percentage}.csv")
    test_file_path = os.path.join(base_path, f"augmented_test_{args.dataset_type}_{args.percentage}.csv")
    val_file_path = os.path.join(base_path, f"augmented_val_{args.dataset_type}_{args.percentage}.csv")

    augmented_train_data = augment_with_metadata_and_topic(original_train_data.copy(), extractor, candidate_labels,
                                                           train_file_path, debug=args.debug)
    augmented_test_data = augment_with_metadata_and_topic(original_test_data.copy(), extractor, candidate_labels,
                                                          test_file_path, debug=args.debug)
    augmented_val_data = augment_with_metadata_and_topic(original_val_data.copy(), extractor, candidate_labels,
                                                         val_file_path, debug=args.debug)

    if args.dataset_type == 'emotion':
        print("Train Data")
        print(augmented_train_data.head())
        print("\nTest Data")
        print(augmented_test_data.head())
        print("\nValidation Data")
        print(augmented_val_data.head())
    elif args.dataset_type == 'sarcasm':
        print("Train Data")
        print(augmented_train_data.head())
        print("\nValidation Data")
        print(augmented_val_data.head())
        print("\nTest Data")
        print(augmented_test_data.head())