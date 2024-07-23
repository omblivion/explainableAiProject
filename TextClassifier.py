from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class TextClassifier:
    def __init__(self, text_column, metadata_columns=[]):
        transformers = [
            ('text', TfidfVectorizer(), text_column)
        ]
        if metadata_columns:
            transformers.append(('metadata', StandardScaler(), metadata_columns))

        # Define a column transformer that handles text and metadata separately
        self.column_transformer = ColumnTransformer(transformers)

        # Define a pipeline with the column transformer and a logistic regression classifier
        self.pipeline = Pipeline([
            ('preprocessor', self.column_transformer),
            ('classifier', LogisticRegression(max_iter=1000))
        ])

    def train(self, X_train, y_train):
        """
        Train the text classifier on the provided training data.

        :param X_train: DataFrame containing training texts and metadata.
        :param y_train: List of training labels.
        """
        self.pipeline.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        """
        Evaluate the text classifier on the provided test data.

        :param X_test: DataFrame containing test texts and metadata.
        :param y_test: List of test labels.
        :return: Classification report as a string.
        """
        y_pred = self.pipeline.predict(X_test)
        report = classification_report(y_test, y_pred)
        return report
