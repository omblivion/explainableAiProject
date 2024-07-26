import os
import pandas as pd
from sklearn.model_selection import train_test_split


class DatasetLoad:
    def __init__(self, dataset_type, base_path, percentage=100.0, debug=False):
        """
        Initialize the DatasetLoad object.

        :param dataset_type: Type of the dataset ('emotion', 'sarcasm', or 'tweets').
        :param base_path: Base path where dataset files are located.
        :param percentage: Percentage of the dataset to use.
        """
        self.dataset_type = dataset_type
        self.base_path = base_path
        self.percentage = percentage
        self.train_data = None
        self.test_data = None
        self.val_data = None

    def load_data(self, file_path):
        """
        Load the tweet dataset from a CSV file.

        :param file_path: Relative path to the tweet dataset file.
        :return: DataFrame containing the tweet data.
        """
        full_path = os.path.join(self.base_path, file_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"File not found: {full_path}")
        data = pd.read_csv(full_path, delimiter=',')
        return data

    def load_datasets(self):
        """
        Load the datasets based on the dataset type and apply percentage sampling if needed.
        Ensure the first column is 'text' and the second column is 'category'.
        """
        if self.dataset_type == 'reddit':
            print("Loading Reddit dataset...")
            data = self.load_data('datasets/Reddit_Data.csv')
            data = data.rename(columns={'clean_comment': 'text'})
            # truncate the text in the text column with over 512 characters
            data['text'] = data['text'].str.slice(0, 512)


        elif self.dataset_type == 'tweets':
            print("Loading Twitter dataset...")
            data = self.load_data('datasets/Twitter_Data.csv')
            # drop the ID column, axis=1
            data = data.drop('Id', axis=1)
            # convert category from text to -1, 0, 1
            data['category'] = data['Category'].map({'negative': 0, 'neutral': 1, 'positive': 2})
            data = data.drop('Category', axis=1)
            # rename 'tweet' column to 'text'
            data = data.rename(columns={'Tweet': 'text'})
            # remove the rows of the text column in which the text is "Not Available"
            data = data[data['text'] != 'Not Available']

        # Ensure the first column is 'text' and the second column is 'category'
        data = data[['text', 'category'] + [col for col in data.columns if col not in ['text', 'category']]]

        train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42)
        self.val_data, self.test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
        self.train_data = train_data

        if self.percentage < 100.0:
            self.train_data = self.train_data.sample(frac=self.percentage / 100.0, random_state=42)
            self.val_data = self.val_data.sample(frac=self.percentage / 100.0, random_state=42)
            self.test_data = self.test_data.sample(frac=self.percentage / 100.0, random_state=42)
