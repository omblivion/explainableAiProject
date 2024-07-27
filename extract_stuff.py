import os

import pandas as pd


def augment_and_extract_metadata(dataset, extractor, topic_labels, file_path, debug=False):
    if os.path.exists(file_path):
        # If the file exists, load the augmented dataset from the CSV file
        print(f"Loading augmented dataset from {file_path}")
        augmented_dataset = pd.read_csv(file_path)
    else:
        # If the file does not exist, proceed with augmenting the dataset
        print(f"Augmenting dataset and saving to {file_path}")
        total_rows = len(dataset)
        count = 0

        topics = []

        # Iterate over each row in the dataset
        for index, row in dataset.iterrows():
            # Extract topic using the extractor
            topic = extractor.extract_topic(row['text'], topic_labels)
            topics.append(topic)

            for label in topic_labels:
                dataset.at[index, label] = 1 if topic == label else 0

            # If debug mode is enabled, print debug information
            percentage_complete = ((count + 1) / total_rows) * 100
            if debug:
                print(f"DEBUG - Text: {row['text']}")
                print(f"DEBUG - Generated Metadata: Topic - {topic}")
                print(f"DEBUG - Percentage of Completion: {percentage_complete:.2f}%, {count + 1} of {total_rows}")

            if int(percentage_complete) % 5 == 0:
                print(f"Percentage of Completion: {percentage_complete:.2f}%, {count + 1} of {total_rows}")

            count += 1

        dataset['topic'] = topics

        dataset.to_csv(file_path, index=False)
        augmented_dataset = dataset

    return augmented_dataset


def predict_sentiment(dataset, sentiment_analyzer, file_path, debug=False, batch_size=32):
    if os.path.exists(file_path):
        # If the file exists, load the sentiment-augmented dataset from the CSV file
        print(f"Loading sentiment-augmented dataset from {file_path}")
        sentiment_augmented_dataset = pd.read_csv(file_path)
    else:
        # If the file does not exist, proceed with sentiment prediction
        print(f"Predicting sentiment and saving to {file_path}")
        total_rows = len(dataset)
        sentiments = []

        # Process the dataset in batches
        for start in range(0, total_rows, batch_size):
            end = min(start + batch_size, total_rows)
            # Extract a batch of texts from the dataset
            batch_texts = dataset['text'][start:end].tolist()

            # Truncate texts to the model's maximum token length
            truncated_batch_texts = [sentiment_analyzer.truncate_text(text) for text in batch_texts]

            batch_results = sentiment_analyzer.classifier(truncated_batch_texts, truncation=True, padding=True,
                                                          max_length=512)
            batch_sentiments = [sentiment_analyzer.map_label_to_target(result['label']) for result in batch_results]
            sentiments.extend(batch_sentiments)

            # Calculate the percentage of completion
            percentage_complete = (end / total_rows) * 100
            if debug:
                print(f"DEBUG - Processed batch {start // batch_size + 1}: {start} to {end}")
                print(f"DEBUG - Percentage of Completion: {percentage_complete:.2f}%, {end} of {total_rows}")
            if int(percentage_complete) % 5 == 0:
                print(f"Percentage of Completion: {percentage_complete:.2f}%")

        dataset['sentiment'] = sentiments
        dataset.to_csv(file_path, index=False)
        sentiment_augmented_dataset = dataset

    return sentiment_augmented_dataset
