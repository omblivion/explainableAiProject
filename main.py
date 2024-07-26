import argparse
import os
import wandb
import pandas as pd
import torch
from sklearn.metrics import classification_report
from DatasetLoad import DatasetLoad
from MetadataExtractor import MetadataExtractor
from SentimentAnalyzer import SentimentAnalyzer
from extract_stuff import augment_and_extract_metadata, predict_sentiment


os.environ["WANDB_API_KEY"] = "21cb0c9433eeca19401ee01e9b1bc9e4b6f7a696"

if __name__ == "__main__":

    # Set up argument parser for command-line options
    parser = argparse.ArgumentParser(description='Load dataset')
    parser.add_argument('--dataset_type', type=str, default='tweets', choices=['tweets', 'TODO'],
                        help='Type of dataset to load')
    parser.add_argument('--debug', type=bool, default=False,
                        help='Enable debug mode to print additional information')
    parser.add_argument('--percentage', type=float, default=100.0,
                        help='Percentage of the dataset to use (e.g., 0.1 for 0.1%)')

    # Parse command-line arguments
    args = parser.parse_args()
    print("Debugging is set to: ", args.debug)
    print("Percentage is set to: ", args.percentage)

    # Print Torch availability and device information
    print("Torch is available: ", torch.cuda.is_available())
    print("Torch device count: ", torch.cuda.device_count())
    print("Torch available device: ", torch.cuda.get_device_name(torch.cuda.current_device()))

    # Initialize dataset loader with the specified type and base path
    base_path = os.path.dirname(os.path.abspath(__file__))
    dataset_loader = DatasetLoad('tweets', base_path, args.percentage)
    dataset_loader.load_datasets()

    # Load the original train, test, and validation datasets
    original_train_data = dataset_loader.train_data
    original_test_data = dataset_loader.test_data
    original_val_data = dataset_loader.val_data

    print(original_train_data.head(5))

    # Initialize the sentiment analyzer
    sentiment_analyzer = SentimentAnalyzer()

    # Fine-tune the sentiment analyzer with the original dataset
    fine_tuning_results = sentiment_analyzer.fine_tune(original_train_data)
    print(f"Fine-tuning results: {fine_tuning_results}")

    # Extract metadata for the datasets
    base_path = os.path.dirname(os.path.abspath(__file__))

    # Extract metadata for the datasets
    train_sentiment_file_name = os.path.join(base_path, f'train_sentiment_{args.dataset_type}_{args.percentage}.csv')
    test_sentiment_file_name = os.path.join(base_path, f'test_sentiment_{args.dataset_type}_{args.percentage}.csv')
    val_sentiment_file_name = os.path.join(base_path, f'val_sentiment_{args.dataset_type}_{args.percentage}.csv')

    # Predict sentiment for the datasets
    train_data_with_sentiment = predict_sentiment(original_train_data.copy(), sentiment_analyzer,
                                                  train_sentiment_file_name, args.debug)
    test_data_with_sentiment = predict_sentiment(original_test_data.copy(), sentiment_analyzer,
                                                 test_sentiment_file_name, args.debug)
    val_data_with_sentiment = predict_sentiment(original_val_data.copy(), sentiment_analyzer, val_sentiment_file_name,
                                                args.debug)


    # Compute metrics for the train dataset
    train_true_labels = original_train_data['category']
    train_predicted_labels = train_data_with_sentiment['sentiment']
    print("\nTrain Classification Report:")
    print(classification_report(train_true_labels, train_predicted_labels, labels=[0, 1, 2], zero_division=0))

    # Compute metrics for the test dataset
    test_true_labels = original_test_data['category']
    test_predicted_labels = test_data_with_sentiment['sentiment']
    print("\nTest Classification Report:")
    print(classification_report(test_true_labels, test_predicted_labels, labels=[0, 1, 2], zero_division=0))

    # Compute metrics for the validation dataset
    val_true_labels = original_val_data['category']
    val_predicted_labels = val_data_with_sentiment['sentiment']
    print("\nValidation Classification Report:")
    print(classification_report(val_true_labels, val_predicted_labels, labels=[0, 1, 2], zero_division=0))


    # Initialize the metadata extractor
    extractor = MetadataExtractor()

    # Define topic labels
    topic_labels = ["politics", "entertainment", "sports", "technology", "health", "education", "finance", "food", "other"]

    # Define the base path where main.py is located
    base_path = os.path.dirname(os.path.abspath(__file__))

    # Extract metadata for the datasets
    train_file_name = os.path.join(base_path, f'train_augmented_{args.dataset_type}_{args.percentage}.csv')
    test_file_name = os.path.join(base_path, f'test_augmented_{args.dataset_type}_{args.percentage}.csv')
    val_file_name = os.path.join(base_path, f'val_augmented_{args.dataset_type}_{args.percentage}.csv')

    train_data_with_metadata = augment_and_extract_metadata(train_data_with_sentiment.copy(), extractor,
                                                            topic_labels, train_file_name, args.debug)
    test_data_with_metadata = augment_and_extract_metadata(test_data_with_sentiment.copy(), extractor,
                                                           topic_labels, test_file_name, args.debug)
    val_data_with_metadata = augment_and_extract_metadata(val_data_with_sentiment.copy(), extractor,
                                                          topic_labels, val_file_name, args.debug)


    # Function to create subgroups based on metadata
    def create_subgroups(dataset):
        subgroups = {}
        for topic in topic_labels:
            subgroup_name = f"{topic}"
            subgroups[subgroup_name] = dataset[dataset['topic'] == topic]
        return subgroups


    # Create subgroups for the datasets
    train_subgroups = create_subgroups(train_data_with_metadata)
    test_subgroups = create_subgroups(test_data_with_metadata)
    val_subgroups = create_subgroups(val_data_with_metadata)

    # Function to compute metrics for the subgroups
    def compute_metrics(subgroups, true_labels_column='category', pred_labels_column='sentiment'):
        metrics = []
        for topic, subgroup in subgroups.items():
            if not subgroup.empty:
                true_labels = subgroup[true_labels_column]
                pred_labels = subgroup[pred_labels_column]
                report = classification_report(true_labels, pred_labels, output_dict=True, zero_division=0)
                metrics.append({
                    'topic': topic,
                    'accuracy': report['accuracy'],
                    'precision': report['weighted avg']['precision'],
                    'recall': report['weighted avg']['recall'],
                    'f1-score': report['weighted avg']['f1-score']
                })
        return pd.DataFrame(metrics)


    train_metrics = compute_metrics(train_subgroups)
    test_metrics = compute_metrics(test_subgroups)
    val_metrics = compute_metrics(val_subgroups)

    print("Train Metrics per Topic")
    print(train_metrics)
    print("\nTest Metrics per Topic")
    print(test_metrics)
    print("\nValidation Metrics per Topic")
    print(val_metrics)

    # Function to analyze disparities in sentiment predictions
    def analyze_disparities(subgroups):
        analysis_results = []
        for subgroup_name, subgroup_data in subgroups.items():
            if not subgroup_data.empty:
                sentiment_counts = subgroup_data['sentiment'].value_counts(normalize=True) * 100
                analysis_results.append({
                    'subgroup': subgroup_name,
                    'total': len(subgroup_data),
                    'negative': sentiment_counts.get(0, 0),
                    'neutral': sentiment_counts.get(1, 0),
                    'positive': sentiment_counts.get(2, 0),
                })
        return pd.DataFrame(analysis_results)


    # Analyze disparities for the datasets
    train_analysis = analyze_disparities(train_subgroups)
    test_analysis = analyze_disparities(test_subgroups)
    val_analysis = analyze_disparities(val_subgroups)

    # Print the analysis results
    print("Train Percentage Analysis")
    print(train_analysis)
    print("\nTest Percentage Analysis")
    print(test_analysis)
    print("\nValidation Percentage Analysis")
    print(val_analysis)

