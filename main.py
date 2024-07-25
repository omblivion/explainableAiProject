import argparse
import os

import pandas as pd
import torch

from DatasetLoad import DatasetLoad
from MetadataExtractor import MetadataExtractor
from SentimentAnalyzer import SentimentAnalyzer
from extract_stuff import augment_and_extract_metadata, predict_sentiment

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

    # Initialize the sentiment analyzer
    sentiment_analyzer = SentimentAnalyzer()

    # Extract metadata for the datasets
    train_sentiment_file_name = f'train_sentiment_{args.dataset_type}_{args.percentage}.csv'
    test_sentiment_file_name = f'test_sentiment_{args.dataset_type}_{args.percentage}.csv'
    val_sentiment_file_name = f'val_sentiment_{args.dataset_type}_{args.percentage}.csv'

    # Predict sentiment for the datasets
    train_data_with_sentiment = predict_sentiment(original_train_data.copy(), sentiment_analyzer,
                                                  train_sentiment_file_name, args.debug)
    test_data_with_sentiment = predict_sentiment(original_test_data.copy(), sentiment_analyzer,
                                                 test_sentiment_file_name, args.debug)
    val_data_with_sentiment = predict_sentiment(original_val_data.copy(), sentiment_analyzer, val_sentiment_file_name,
                                                args.debug)

    # Initialize the metadata extractor
    extractor = MetadataExtractor()

    # Define gender and topic labels
    gender_labels = ["male", "female", "unknown"]
    topic_labels = ["politics", "news", "entertainment", "unknown"]

    # Extract metadata for the datasets
    train_file_name = f'train_augmented_{args.dataset_type}_{args.percentage}.csv'
    test_file_name = f'test_augmented_{args.dataset_type}_{args.percentage}.csv'
    val_file_name = f'val_augmented_{args.dataset_type}_{args.percentage}.csv'

    train_data_with_metadata = augment_and_extract_metadata(train_data_with_sentiment.copy(), extractor, gender_labels,
                                                            topic_labels, train_file_name, args.debug)
    test_data_with_metadata = augment_and_extract_metadata(test_data_with_sentiment.copy(), extractor, gender_labels,
                                                           topic_labels, test_file_name, args.debug)
    val_data_with_metadata = augment_and_extract_metadata(val_data_with_sentiment.copy(), extractor, gender_labels,
                                                          topic_labels, val_file_name, args.debug)


    # Function to create subgroups based on metadata
    def create_subgroups(dataset):
        subgroups = {}
        for gender in gender_labels:
            for topic in topic_labels:
                subgroup_name = f"{gender}_{topic}"
                subgroups[subgroup_name] = dataset[(dataset['gender'] == gender) & (dataset['topic'] == topic)]
        return subgroups


    # Create subgroups for the datasets
    train_subgroups = create_subgroups(train_data_with_metadata)
    test_subgroups = create_subgroups(test_data_with_metadata)
    val_subgroups = create_subgroups(val_data_with_metadata)


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
                    'neutral': sentiment_counts.get(2, 0),
                    'positive': sentiment_counts.get(4, 0),
                })
        return pd.DataFrame(analysis_results)


    # Analyze disparities for the datasets
    train_analysis = analyze_disparities(train_subgroups)
    test_analysis = analyze_disparities(test_subgroups)
    val_analysis = analyze_disparities(val_subgroups)

    # Save the analysis results to CSV files
    # train_analysis.to_csv('train_sentiment_analysis.csv', index=False)
    # test_analysis.to_csv('test_sentiment_analysis.csv', index=False)
    # val_analysis.to_csv('val_sentiment_analysis.csv', index=False)

    # Print the analysis results
    print("Train Sentiment Analysis")
    print(train_analysis)
    print("\nTest Sentiment Analysis")
    print(test_analysis)
    print("\nValidation Sentiment Analysis")
    print(val_analysis)

    # Function to analyze disparities in sentiment predictions
    # # Initialize the sentiment analyzer
    # sentiment_analyzer = SentimentAnalyzer()
    #
    # # Analyze sentiment for the datasets
    # for dataset in [augmented_train_data, augmented_test_data, augmented_val_data]:
    #     for index, row in dataset.iterrows():
    #         sentiment_label = sentiment_analyzer.analyze_sentiment(row['text'])
    #         sentiment_target = sentiment_analyzer.map_label_to_target(sentiment_label)
    #         dataset.at[index, 'sentiment'] = sentiment_target
