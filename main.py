import numpy as np
from DatasetLoad import DatasetLoad
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import argparse
from MetadataExtractor import MetadataExtractor


def augment_with_metadata_and_topic(dataset, extractor, candidate_labels, debug=False):
    """
    Augments the dataset with metadata and the probabilities for each candidate label.

    :param dataset: A pandas DataFrame containing the dataset.
    :param extractor: An instance of the MetadataExtractor class.
    :param candidate_labels: A list of strings representing candidate topics.
    :return: The augmented dataset with probabilities for each candidate label.
    """
    total_rows = len(dataset)
    count = 0 # Initialize the count variable
    for index, row in dataset.iterrows():
        probabilities = extractor.extract_probabilities(row['text'], candidate_labels)
        # Update the dataset with the new metadata
        for label in candidate_labels:
            dataset.at[index, label] = probabilities.get(label, 0)

        if debug:
            # Calculate and print the percentage of completion
            percentage_complete = ((count+1) / total_rows) * 100
            print(f"Text: {row['text']}")
            print("Generated Metadata:", probabilities)
            print(f"Percentage of Completion: {percentage_complete:.2f}%, {count+1} of {total_rows}")

        count += 1

    return dataset


# Step 3: Load and Augment Datasets
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

    dataset_loader = DatasetLoad(args.dataset_type, percentage=args.percentage)
    dataset_loader.load_datasets()

    extractor = MetadataExtractor()
    candidate_labels = ["happiness", "sadness", "anger", "fear", "love", "surprise"]

    original_train_data = dataset_loader.train_data
    original_test_data = dataset_loader.test_data
    original_val_data = dataset_loader.val_data

    augmented_train_data = augment_with_metadata_and_topic(original_train_data.copy(), extractor, candidate_labels, debug=args.debug)
    augmented_test_data = augment_with_metadata_and_topic(original_test_data.copy(), extractor, candidate_labels, debug=args.debug)
    augmented_val_data = augment_with_metadata_and_topic(original_val_data.copy(), extractor, candidate_labels, debug=args.debug)

    if args.dataset_type == 'emotion':
        print("Train Data")
        print(dataset_loader.train_data.head())
        print("\nTest Data")
        print(dataset_loader.test_data.head())
        print("\nValidation Data")
        print(dataset_loader.val_data.head())
    elif args.dataset_type == 'sarcasm':
        print("Train Data")
        print(dataset_loader.train_data.head())
        print("\nValidation Data")
        print(dataset_loader.val_data.head())
        print("\nTest Data")
        print(dataset_loader.test_data.head())