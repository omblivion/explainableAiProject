import os

import torch

from DatasetLoad import DatasetLoad
from MetadataExtractor import MetadataExtractor
from TextClassifier import TextClassifier
from dataset_agumentation import augment_with_metadata_and_topic
from utils import print_demographic_distribution


def load_datasets(dataset_type, base_path, percentage):
    """
    Load the train, test, and validation datasets based on the dataset type.

    :param dataset_type: Type of the dataset ('emotion' or 'sarcasm').
    :param base_path: Base path where dataset files are located.
    :param percentage: Percentage of the dataset to use.
    :return: Tuple containing the train, test, and validation datasets.
    """
    dataset_loader = DatasetLoad(dataset_type, base_path, percentage=percentage)
    dataset_loader.load_datasets()
    return dataset_loader.train_data, dataset_loader.test_data, dataset_loader.val_data


def augment_datasets(train_data, test_data, val_data, extractor, base_path, dataset_type, percentage, debug):
    """
    Augment the datasets with metadata and save them to CSV files.

    :param train_data: Training dataset.
    :param test_data: Test dataset.
    :param val_data: Validation dataset.
    :param extractor: Metadata extractor instance.
    :param base_path: Base path where augmented files will be saved.
    :param dataset_type: Type of the dataset ('emotion' or 'sarcasm').
    :param percentage: Percentage of the dataset to use.
    :param debug: Flag to enable debug mode.
    :return: Tuple containing the augmented train, test, and validation datasets.
    """
    train_file_path = os.path.join(base_path, f"augmented_train_{dataset_type}_{percentage}.csv")
    test_file_path = os.path.join(base_path, f"augmented_test_{dataset_type}_{percentage}.csv")
    val_file_path = os.path.join(base_path, f"augmented_val_{dataset_type}_{percentage}.csv")

    augmented_train_data = augment_with_metadata_and_topic(train_data.copy(), extractor, train_file_path, debug=debug)
    augmented_test_data = augment_with_metadata_and_topic(test_data.copy(), extractor, test_file_path, debug=debug)
    augmented_val_data = augment_with_metadata_and_topic(val_data.copy(), extractor, val_file_path, debug=debug)

    return augmented_train_data, augmented_test_data, augmented_val_data


def prepare_data(data, text_column='text', label_column='label', metadata_columns=[]):
    """
    Prepare the data by selecting the text and metadata columns.

    :param data: DataFrame containing the data.
    :param text_column: Name of the text column.
    :param label_column: Name of the label column.
    :param metadata_columns: List of metadata columns.
    :return: Tuple containing the features (X) and labels (y).
    """
    X = data[[text_column] + metadata_columns]
    y = data[label_column]
    return X, y


def train_and_evaluate_classifier(X_train, y_train, X_test, y_test, X_val, y_val, text_column, metadata_columns,
                                  param_grid=None):
    classifier = TextClassifier(text_column, metadata_columns, param_grid=param_grid)
    classifier.train(X_train, y_train)

    print("Evaluation on Test Data:")
    test_report = classifier.evaluate(X_test, y_test)
    print(test_report['classification_report'])
    print(f"Accuracy: {test_report['accuracy']}")
    print(f"F1 Score: {test_report['f1_score']}")
    print(f"Precision: {test_report['precision']}")
    print(f"Recall: {test_report['recall']}")

    print("Evaluation on Validation Data:")
    val_report = classifier.evaluate(X_val, y_val)
    print(val_report['classification_report'])
    print(f"Accuracy: {val_report['accuracy']}")
    print(f"F1 Score: {val_report['f1_score']}")
    print(f"Precision: {val_report['precision']}")
    print(f"Recall: {val_report['recall']}")


def main(args):
    """
    Main function to load datasets, augment them, train and evaluate classifiers.

    :param args: Command-line arguments.
    """
    print("Debugging is set to: ", args.debug)
    print("Percentage is set to: ", args.percentage)
    print("Torch is available: ", torch.cuda.is_available())
    print("Torch device count: ", torch.cuda.device_count())
    print("Torch available device: ", torch.cuda.get_device_name(torch.cuda.current_device()))

    base_path = os.path.dirname(os.path.abspath(__file__))

    # Load datasets
    train_data, test_data, val_data = load_datasets(args.dataset_type, base_path, args.percentage)
    extractor = MetadataExtractor()

    # Augment datasets with metadata
    augmented_train_data, augmented_test_data, augmented_val_data = augment_datasets(
        train_data, test_data, val_data, extractor, base_path, args.dataset_type, args.percentage, args.debug
    )

    # Print demographic distribution of the augmented datasets
    print("Train Data")
    print_demographic_distribution(augmented_train_data)
    print("\nValidation Data")
    print_demographic_distribution(augmented_val_data)
    print("\nTest Data")
    print_demographic_distribution(augmented_test_data)

    # Prepare data for the original classifier
    X_train_orig, y_train_orig = prepare_data(train_data)
    X_test_orig, y_test_orig = prepare_data(test_data)
    X_val_orig, y_val_orig = prepare_data(val_data)

    # Train and evaluate the original classifier
    print("Training and evaluating the original classifier")
    train_and_evaluate_classifier(X_train_orig, y_train_orig, X_test_orig, y_test_orig, X_val_orig, y_val_orig, 'text',
                                  [])

    # Define metadata columns
    gender_labels = ["male", "female", "unknown"]
    race_labels = ["white", "black", "asian", "hispanic", "other", "non-identified"]
    metadata_columns = gender_labels + race_labels

    # Prepare data for the metadata-included classifier
    X_train_meta, y_train_meta = prepare_data(augmented_train_data, metadata_columns=metadata_columns)
    X_test_meta, y_test_meta = prepare_data(augmented_test_data, metadata_columns=metadata_columns)
    X_val_meta, y_val_meta = prepare_data(augmented_val_data, metadata_columns=metadata_columns)

    # Train and evaluate the metadata-included classifier
    print("Training and evaluating the metadata-included classifier")
    param_grid = {
        'classifier__C': [0.1, 1, 10],
        'classifier__solver': ['lbfgs', 'liblinear']
    }
    train_and_evaluate_classifier(X_train_meta, y_train_meta, X_test_meta, y_test_meta, X_val_meta, y_val_meta, 'text',
                                  metadata_columns, param_grid)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Load dataset')
    parser.add_argument('--dataset_type', type=str, default='emotion', choices=['emotion', 'sarcasm'],
                        help='Type of dataset to load')
    parser.add_argument('--debug', type=bool, default=False,
                        help='Enable debug mode to print additional information')
    parser.add_argument('--percentage', type=float, default=100.0,
                        help='Percentage of the dataset to use (e.g., 0.1 for 0.1%)')

    args = parser.parse_args()
    main(args)
