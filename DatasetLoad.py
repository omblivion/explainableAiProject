import json
import os

import pandas as pd
from sklearn.model_selection import train_test_split


class DatasetLoad:
    def __init__(self, dataset_type, base_path, percentage=100.0):
        """
        Initialize the DatasetLoad object.

        :param dataset_type: Type of the dataset ('emotion' or 'sarcasm').
        :param base_path: Base path where dataset files are located.
        :param percentage: Percentage of the dataset to use.
        """
        self.dataset_type = dataset_type
        self.base_path = base_path
        self.percentage = percentage
        self.train_data = None
        self.test_data = None
        self.val_data = None

    def load_emotion_data(self, file_path):
        """
        Load the emotion dataset from a file.

        :param file_path: Relative path to the emotion dataset file.
        :return: DataFrame containing the emotion data.
        """
        full_path = os.path.join(self.base_path, file_path)  # Construct the full file path
        if not os.path.exists(full_path):  # Check if the file exists
            raise FileNotFoundError(f"File not found: {full_path}")
        data = pd.read_csv(full_path, delimiter=';', header=None, names=['text', 'label'])  # Read the CSV file
        return data

    def load_sarcasm_data(self, file_path):
        """
        Load the sarcasm dataset from a JSON file.

        :param file_path: Relative path to the sarcasm dataset file.
        :return: Tuple of DataFrames containing the train, validation, and test data.
        """
        full_path = os.path.join(self.base_path, file_path)  # Construct the full file path
        if not os.path.exists(full_path):  # Check if the file exists
            raise FileNotFoundError(f"File not found: {full_path}")
        with open(full_path, 'r') as f:  # Open the JSON file
            data = [json.loads(line) for line in f]  # Load each line as a JSON object
        df = pd.DataFrame(data)  # Convert the list of JSON objects to a DataFrame

        # Split the data into training, validation, and test sets
        train_data, temp_data = train_test_split(df, test_size=0.3, random_state=42)
        val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

        return train_data, val_data, test_data

    def load_datasets(self):
        """
        Load the datasets based on the dataset type and apply percentage sampling if needed.
        """
        if self.dataset_type == 'emotion':
            # Load the emotion dataset
            self.train_data = self.load_emotion_data('datasets/emotion_NLP_Dataset/train.txt')
            self.test_data = self.load_emotion_data('datasets/emotion_NLP_Dataset/test.txt')
            self.val_data = self.load_emotion_data('datasets/emotion_NLP_Dataset/val.txt')
        elif self.dataset_type == 'sarcasm':
            # Load the sarcasm dataset
            self.train_data, self.val_data, self.test_data = self.load_sarcasm_data(
                'datasets/sarcasm_headlines/Sarcasm_Headlines_Dataset_v2.json')

        # Apply percentage sampling if the percentage is less than 100%
        if self.percentage < 100.0:
            self.train_data = self.train_data.sample(frac=self.percentage / 100.0, random_state=42)
            self.val_data = self.val_data.sample(frac=self.percentage / 100.0, random_state=42)
            self.test_data = self.test_data.sample(frac=self.percentage / 100.0, random_state=42)
