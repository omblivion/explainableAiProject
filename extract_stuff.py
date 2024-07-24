import os

import pandas as pd


def augment_and_extract_metadata(dataset, extractor, gender_labels, topic_labels, file_path, debug=False):
    if os.path.exists(file_path):
        # If the file exists, load the augmented dataset from the CSV file
        print(f"Loading augmented dataset from {file_path}")
        augmented_dataset = pd.read_csv(file_path)
    else:
        # If the file does not exist, proceed with augmenting the dataset
        print(f"Augmenting dataset and saving to {file_path}")
        total_rows = len(dataset)
        count = 0

        genders = []
        topics = []

        # Iterate over each row in the dataset
        for index, row in dataset.iterrows():
            # Extract gender and topic using the extractor
            gender = extractor.extract_gender(row['text'], gender_labels)
            topic = extractor.extract_topic(row['text'], topic_labels)

            genders.append(gender)
            topics.append(topic)

            for label in gender_labels:
                dataset.at[index, label] = 1 if gender == label else 0

            for label in topic_labels:
                dataset.at[index, label] = 1 if topic == label else 0

            # If debug mode is enabled, print debug information
            percentage_complete = ((count + 1) / total_rows) * 100
            if debug:
                print(f"Text: {row['text']}")
                print(f"Generated Metadata: Gender - {gender}, Topic - {topic}")
                print(f"Percentage of Completion: {percentage_complete:.2f}%, {count + 1} of {total_rows}")

            if percentage_complete % 5 == 0:
                print(f"Percentage of Completion: {percentage_complete:.2f}%, {count + 1} of {total_rows}")

            count += 1

        dataset['gender'] = genders
        dataset['topic'] = topics

        dataset.to_csv(file_path, index=False)
        augmented_dataset = dataset

    return augmented_dataset


# Function to predict sentiment for a dataset
def predict_sentiment(dataset, sentimentAnalyzer, debug=False):
    sentiments = []
    total_rows = len(dataset)

    count = 0
    for index, row in dataset.iterrows():
        sentiment_label = sentimentAnalyzer.analyze_sentiment(row['text'])
        sentiment_target = sentimentAnalyzer.map_label_to_target(sentiment_label)
        sentiments.append(sentiment_target)
        count += 1

        # Calculate the percentage of completion
        percentage_complete = ((count + 1) / total_rows) * 100
        if debug:
            print(f"Text: {row['text']} Sentiment - {sentiment_label}")
            print(f"Percentage of Completion: {percentage_complete:.2f}%, {count + 1} of {total_rows}")
        if percentage_complete % 5 == 0:
            print(f"Percentage of Completion: {percentage_complete:.2f}%, {count + 1} of {total_rows}")

    dataset['sentiment'] = sentiments
    return dataset
