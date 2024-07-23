import json
import pandas as pd
from sklearn.model_selection import train_test_split

class DatasetLoad:
    def __init__(self, dataset_type, percentage=100.0):
        self.dataset_type = dataset_type
        self.percentage = percentage
        self.train_data = None
        self.test_data = None
        self.val_data = None

    def load_emotion_data(self, file_path):
        data = pd.read_csv(file_path, delimiter=';', header=None, names=['text', 'label'])
        return data

    def load_sarcasm_data(self, file_path):
        with open(file_path, 'r') as f:
            data = [json.loads(line) for line in f]
        df = pd.DataFrame(data)

        # Split the data
        train_data, temp_data = train_test_split(df, test_size=0.3, random_state=42)
        val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

        return train_data, val_data, test_data

    def load_datasets(self):
        if self.dataset_type == 'emotion':
            self.train_data = self.load_emotion_data('datasets/emotion_NLP_Dataset/train.txt')
            self.test_data = self.load_emotion_data('datasets/emotion_NLP_Dataset/test.txt')
            self.val_data = self.load_emotion_data('datasets/emotion_NLP_Dataset/val.txt')
        elif self.dataset_type == 'sarcasm':
            self.train_data, self.val_data, self.test_data = self.load_sarcasm_data(
                'datasets/sarcasm_headlines/Sarcasm_Headlines_Dataset_v2.json')

        if self.percentage < 100.0:
            self.train_data = self.train_data.sample(frac=self.percentage / 100.0, random_state=42)
            self.val_data = self.val_data.sample(frac=self.percentage / 100.0, random_state=42)
            self.test_data = self.test_data.sample(frac=self.percentage / 100.0, random_state=42)

