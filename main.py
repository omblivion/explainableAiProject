import argparse
import os

import matplotlib.pyplot as plt
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
                        help='Enable debug mode to print even more additional information')
    parser.add_argument('--deep_debug', type=bool, default=False,
                        help='Enable debug mode to print additional information')
    parser.add_argument('--percentage', type=float, default=100.0,
                        help='Percentage of the dataset to use (e.g., 0.1 for 0.1%)')

    # Parse command-line arguments
    args = parser.parse_args()
    print("Debugging is set to: ", args.debug)
    print("Deep Debugging is set to: ", args.deep_debug)
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

    # Extract metadata for the datasets
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_save_path = os.path.join(base_path, f'sentiment_model_{args.dataset_type}_{args.percentage}.pt')
    # Check if a saved model exists
    if os.path.exists(model_save_path):
        print("Loading the fine-tuned model from disk...")
        sentiment_analyzer.model = torch.load(model_save_path)
    else:
        print("Fine-tuning the sentiment analyzer with the original dataset...")
        fine_tuning_results = sentiment_analyzer.fine_tune(original_train_data)
        print(f"Fine-tuning results: {fine_tuning_results}")
        # Save the fine-tuned model
        torch.save(sentiment_analyzer.model, model_save_path)

    # Define the file names for the sentiment predictions
    train_sentiment_file_name = os.path.join(base_path, f'train_sentiment_{args.dataset_type}_{args.percentage}.csv')
    test_sentiment_file_name = os.path.join(base_path, f'test_sentiment_{args.dataset_type}_{args.percentage}.csv')
    val_sentiment_file_name = os.path.join(base_path, f'val_sentiment_{args.dataset_type}_{args.percentage}.csv')

    # Predict sentiment for the datasets
    train_data_with_sentiment = predict_sentiment(original_train_data.copy(), sentiment_analyzer,
                                                  train_sentiment_file_name, args.deep_debug)
    test_data_with_sentiment = predict_sentiment(original_test_data.copy(), sentiment_analyzer,
                                                 test_sentiment_file_name, args.deep_debug)
    val_data_with_sentiment = predict_sentiment(original_val_data.copy(), sentiment_analyzer, val_sentiment_file_name,
                                                args.deep_debug)


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
    topic_labels = ["news", "entertainment", "sports", "technology", "health", "education", "business", "lifestyle", "opinions", "other"]

    # Define the base path where main.py is located
    base_path = os.path.dirname(os.path.abspath(__file__))

    # Extract metadata for the datasets
    train_file_name = os.path.join(base_path, f'train_augmented_{args.dataset_type}_{args.percentage}.csv')
    test_file_name = os.path.join(base_path, f'test_augmented_{args.dataset_type}_{args.percentage}.csv')
    val_file_name = os.path.join(base_path, f'val_augmented_{args.dataset_type}_{args.percentage}.csv')

    train_data_with_metadata = augment_and_extract_metadata(train_data_with_sentiment.copy(), extractor,
                                                            topic_labels, train_file_name, args.deep_debug)
    test_data_with_metadata = augment_and_extract_metadata(test_data_with_sentiment.copy(), extractor,
                                                           topic_labels, test_file_name, args.deep_debug)
    val_data_with_metadata = augment_and_extract_metadata(val_data_with_sentiment.copy(), extractor,
                                                          topic_labels, val_file_name, args.deep_debug)


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


    def weighted_metrics(metrics_df, support_df, metric='accuracy'):
        # Join metrics with their respective support counts
        metrics_df = metrics_df.copy()
        metrics_df = metrics_df.merge(support_df, left_on='topic', right_on='subgroup')
        metrics_df['weighted_metric'] = metrics_df[metric] * metrics_df['total']
        return metrics_df


    def get_top_lower_topics(test_metrics_df, test_percentage_analysis_df, metric='accuracy'):
        # Get support for each topic
        support_df = test_percentage_analysis_df[['subgroup', 'total']].rename(columns={'total': 'support'})

        # Compute weighted metrics
        weighted_metrics_df = weighted_metrics(test_metrics_df, support_df, metric)

        # Compute baseline accuracy
        baseline_accuracy = weighted_metrics_df['accuracy'].mean()

        # Sort topics by their weighted metrics
        sorted_metrics = weighted_metrics_df.sort_values(by='weighted_metric', ascending=False) # Sort by descending

        # Get top 3 and bottom 3 topics
        top_3_topics = sorted_metrics.head(3)['topic'].tolist()
        bottom_3_topics = sorted_metrics.tail(3)['topic'].tolist()

        # Adjust for baseline accuracy
        bottom_3_topics_below_baseline = sorted_metrics[sorted_metrics['accuracy'] < baseline_accuracy].tail(3)[
            'topic'].tolist()

        return top_3_topics, bottom_3_topics_below_baseline


    top_3_topics, bottom_3_topics = get_top_lower_topics(val_metrics, val_analysis, metric='accuracy')
    print(f"Bottom 3 validation topics: {bottom_3_topics }")

    print("Augmenting the training dataset with synthetic data...")
    # Randomly select rows from bottom three topics in the training set
    train_data_bottom_3 = train_data_with_metadata[train_data_with_metadata['topic'].isin(bottom_3_topics)]
    selected_samples = train_data_bottom_3.sample(n=50, random_state=42)    # Select n samples from the bottom 3 topics

    # Augment the selected samples using the sentiment analyzer
    generated_df, generated_df_with_metadata = sentiment_analyzer.generate_training_data(
        selected_samples['topic'].tolist(),
        selected_samples['text'].tolist(),
        selected_samples['sentiment'].tolist(),
        debug=args.debug
    )

    # Combine the original and augmented datasets
    train_original_and_generated_data = pd.concat([original_train_data, generated_df], ignore_index=True)
    # Save the combined datasets
    train_original_and_generated_data.to_csv(os.path.join(base_path, 'train_original_and_generated_data.csv'), index=False)

    model_save_path_v2 = os.path.join(base_path, f'finetuned_sentiment_model_{args.dataset_type}_{args.percentage}.pt')
    print("Fine-tuning the sentiment analyzer with the generated+original dataset...")
    fine_tuning_results_new = sentiment_analyzer.fine_tune(train_original_and_generated_data)  # TODO NON CE
    print(f"Fine-tuning results: {fine_tuning_results_new}")
    # Save the fine-tuned model
    torch.save(sentiment_analyzer.model, model_save_path_v2)

    # Predict sentiment for the original dataset to see for improvements
    test_sentiment_file_name_v2 = os.path.join(base_path, f'test_sentiment_v2_{args.dataset_type}_{args.percentage}.csv')
    val_sentiment_file_name_v2 = os.path.join(base_path, f'val_sentiment_v2_{args.dataset_type}_{args.percentage}.csv')
    test_data_with_sentiment_v2 = predict_sentiment(original_test_data.copy(), sentiment_analyzer,
                                                    test_sentiment_file_name_v2, args.deep_debug)
    val_data_with_sentiment_v2 = predict_sentiment(original_val_data.copy(), sentiment_analyzer,
                                                   val_sentiment_file_name_v2, args.deep_debug)

    # Compute metrics for the test dataset
    test_true_labels = original_test_data['category']
    test_predicted_labels_v2 = test_data_with_sentiment_v2['sentiment']
    print("\nTest Classification Report:")
    print(classification_report(test_true_labels, test_predicted_labels_v2, labels=[0, 1, 2], zero_division=0))

    # Compute metrics for the validation dataset
    val_true_labels = original_val_data['category']
    val_predicted_labels_v2 = val_data_with_sentiment_v2['sentiment']
    print("\nValidation Classification Report:")
    print(classification_report(val_true_labels, val_predicted_labels_v2, labels=[0, 1, 2], zero_division=0))

    test_file_name_v2 = os.path.join(base_path, f'test_augmented_v2_{args.dataset_type}_{args.percentage}.csv')
    val_file_name_v2 = os.path.join(base_path, f'val_augmented_v2_{args.dataset_type}_{args.percentage}.csv')
    test_data_with_metadata_v2 = augment_and_extract_metadata(test_data_with_sentiment_v2.copy(), extractor,
                                                              topic_labels, test_file_name_v2, args.deep_debug)
    val_data_with_metadata_v2 = augment_and_extract_metadata(val_data_with_sentiment_v2.copy(), extractor, topic_labels,
                                                             val_file_name_v2, args.deep_debug)

    # Create subgroups for the datasets
    test_subgroups_v2 = create_subgroups(test_data_with_metadata_v2)
    val_subgroups_v2 = create_subgroups(val_data_with_metadata_v2)


    test_metrics_v2 = compute_metrics(test_subgroups_v2)
    val_metrics_v2 = compute_metrics(val_subgroups_v2)

    print("\nTest Metrics per Topic")
    print(test_metrics_v2)
    print("\nValidation Metrics per Topic")
    print(val_metrics_v2)

    test_analysis_v2 = analyze_disparities(test_subgroups_v2)
    val_analysis_v2 = analyze_disparities(val_subgroups_v2)
    print("\nTest Percentage Analysis")
    print(test_analysis_v2)
    print("\nValidation Percentage Analysis")
    print(val_analysis_v2)


    def plot_metrics_comparison(old_metrics, new_metrics, metric='accuracy'):
        """
        Plots a comparison of the given metric before and after fine-tuning.

        Parameters:
        - old_metrics: DataFrame containing the old metrics
        - new_metrics: DataFrame containing the new metrics
        - metric: The metric to compare (default is 'accuracy')
        """
        # Merge the old and new metrics on the 'topic' column
        comparison_df = old_metrics.merge(new_metrics, on='topic', suffixes=('_old', '_new'))

        # Sort the DataFrame by the new metric for better visualization
        comparison_df = comparison_df.sort_values(by=f'{metric}_new', ascending=False)

        # Plot the comparison
        plt.figure(figsize=(12, 8))
        bar_width = 0.4

        # Positioning the bars
        r1 = range(len(comparison_df))
        r2 = [x + bar_width for x in r1]

        plt.bar(r1, comparison_df[f'{metric}_old'], color='blue', width=bar_width, edgecolor='grey', label='Old')
        plt.bar(r2, comparison_df[f'{metric}_new'], color='green', width=bar_width, edgecolor='grey', label='New')

        plt.xlabel('Topics', fontweight='bold')
        plt.ylabel(metric.capitalize(), fontweight='bold')
        plt.title(f'Comparison of {metric.capitalize()} by Topic', fontweight='bold')
        plt.xticks([r + bar_width / 2 for r in range(len(comparison_df))], comparison_df['topic'], rotation=90)
        plt.legend()

        plt.tight_layout()
        plt.show()


    # Plot the comparison for accuracy, precision, recall, and f1-score
    plot_metrics_comparison(test_metrics, test_metrics_v2, metric='accuracy')
    plot_metrics_comparison(test_metrics, test_metrics_v2, metric='precision')
    plot_metrics_comparison(test_metrics, test_metrics_v2, metric='recall')
    plot_metrics_comparison(test_metrics, test_metrics_v2, metric='f1-score')


    def calculate_overall_accuracy(metrics_df):
        """
        Calculate the overall accuracy from the metrics DataFrame.

        Parameters:
        - metrics_df: DataFrame containing the metrics

        Returns:
        - overall_accuracy: The overall accuracy
        """
        total_support = metrics_df['total'].sum()
        weighted_accuracy_sum = (metrics_df['accuracy'] * metrics_df['total']).sum()
        overall_accuracy = weighted_accuracy_sum / total_support
        return overall_accuracy


    The
    error
    you
    're encountering, KeyError: '
    total
    ', indicates that the DataFrame metrics_df does not contain a column named '
    total
    '. This likely happens in the function calculate_overall_accuracy.

    To
    resolve
    this, you
    need
    to
    ensure
    that
    the
    DataFrame
    passed
    to
    calculate_overall_accuracy
    has
    a
    'total'
    column.The
    'total'
    column
    appears
    to
    represent
    the
    support(i.e., the
    count
    of
    instances) for each topic in your analysis.This support data should be extracted from the analyze_disparities function.

    Here’s
    a
    modified
    version
    of
    the
    relevant
    functions and parts
    of
    your
    code, ensuring
    that
    the
    'total'
    column is present in the
    DataFrame
    passed
    to
    calculate_overall_accuracy.

    python


    def weighted_metrics(metrics_df, support_df, metric='accuracy'):
        # Join metrics with their respective support counts
        metrics_df = metrics_df.copy()
        metrics_df = metrics_df.merge(support_df, left_on='topic', right_on='subgroup')
        metrics_df['weighted_metric'] = metrics_df[metric] * metrics_df['total']
        return metrics_df


    def get_top_lower_topics(test_metrics_df, test_percentage_analysis_df, metric='accuracy'):
        # Get support for each topic
        support_df = test_percentage_analysis_df[['subgroup', 'total']].rename(columns={'total': 'support'})

        # Compute weighted metrics
        weighted_metrics_df = weighted_metrics(test_metrics_df, support_df, metric)

        # Compute baseline accuracy
        baseline_accuracy = weighted_metrics_df['accuracy'].mean()

        # Sort topics by their weighted metrics
        sorted_metrics = weighted_metrics_df.sort_values(by='weighted_metric', ascending=False)  # Sort by descending

        # Get top 3 and bottom 3 topics
        top_3_topics = sorted_metrics.head(3)['topic'].tolist()
        bottom_3_topics = sorted_metrics.tail(3)['topic'].tolist()

        # Adjust for baseline accuracy
        bottom_3_topics_below_baseline = sorted_metrics[sorted_metrics['accuracy'] < baseline_accuracy].tail(3)[
            'topic'].tolist()

        return top_3_topics, bottom_3_topics_below_baseline


    def calculate_overall_accuracy(metrics_df):
        """
        Calculate the overall accuracy from the metrics DataFrame.

        Parameters:
        - metrics_df: DataFrame containing the metrics

        Returns:
        - overall_accuracy: The overall accuracy
        """
        total_support = metrics_df['total'].sum()
        weighted_accuracy_sum = (metrics_df['accuracy'] * metrics_df['total']).sum()
        overall_accuracy = weighted_accuracy_sum / total_support
        return overall_accuracy


    def plot_overall_accuracy_comparison(old_metrics, new_metrics, support_old, support_new):
        """
        Plot the overall accuracy comparison before and after fine-tuning.

        Parameters:
        - old_metrics: DataFrame containing the old metrics
        - new_metrics: DataFrame containing the new metrics
        - support_old: DataFrame containing the support data for the old metrics
        - support_new: DataFrame containing the support data for the new metrics
        """
        old_metrics_with_total = old_metrics.merge(support_old[['subgroup', 'total']], left_on='topic',
                                                   right_on='subgroup')
        new_metrics_with_total = new_metrics.merge(support_new[['subgroup', 'total']], left_on='topic',
                                                   right_on='subgroup')

        overall_accuracy_old = calculate_overall_accuracy(old_metrics_with_total)
        overall_accuracy_new = calculate_overall_accuracy(new_metrics_with_total)

        accuracies = [overall_accuracy_old, overall_accuracy_new]
        labels = ['Old Model', 'New Model']

        plt.figure(figsize=(8, 6))
        plt.bar(labels, accuracies, color=['blue', 'green'], edgecolor='grey')

        plt.xlabel('Model', fontweight='bold')
        plt.ylabel('Overall Accuracy', fontweight='bold')
        plt.title('Overall Accuracy Comparison', fontweight='bold')
        plt.ylim(0, 1)  # Assuming accuracy is between 0 and 1

        for i, v in enumerate(accuracies):
            plt.text(i, v + 0.01, f"{v:.2f}", ha='center', fontweight='bold')

        plt.tight_layout()
        plt.show()


    # Ensure the 'total' column exists in the metrics DataFrame
    test_analysis_v2 = analyze_disparities(test_subgroups_v2)
    val_analysis_v2 = analyze_disparities(val_subgroups_v2)

    # Print the DataFrame to verify the 'total' column
    print("\nTest Analysis V2:")
    print(test_analysis_v2)
    print("\nValidation Analysis V2:")
    print(val_analysis_v2)

    # Calculate and plot the overall accuracy comparison
    plot_overall_accuracy_comparison(test_metrics, test_metrics_v2, test_analysis, test_analysis_v2)