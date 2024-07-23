from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class TextClassifier:
    def __init__(self, text_column, metadata_columns=[], model=None, param_grid=None):
        # Define the default model if none provided
        if model is None:
            model = LogisticRegression(max_iter=1000)

        # Define the transformers for text and metadata
        transformers = [
            ('text', Pipeline([
                ('vectorizer', TfidfVectorizer())
            ]), text_column)
        ]

        if metadata_columns:
            transformers.append(('metadata', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('scaler', StandardScaler())
            ]), metadata_columns))

        # Define a column transformer that handles text and metadata separately
        self.column_transformer = ColumnTransformer(transformers)

        # Define a pipeline with the column transformer and the classifier
        self.pipeline = Pipeline([
            ('preprocessor', self.column_transformer),
            ('classifier', model)
        ])

        # Define the parameter grid for hyperparameter tuning
        self.param_grid = param_grid

    def train(self, X_train, y_train):
        """
        Train the text classifier on the provided training data.

        :param X_train: DataFrame containing training texts and metadata.
        :param y_train: List of training labels.
        """
        if self.param_grid:
            self.grid_search = GridSearchCV(self.pipeline, self.param_grid, cv=5, scoring='accuracy')
            self.grid_search.fit(X_train, y_train)
            self.best_model = self.grid_search.best_estimator_
        else:
            self.pipeline.fit(X_train, y_train)
            self.best_model = self.pipeline

    def evaluate(self, X_test, y_test):
        """
        Evaluate the text classifier on the provided test data.

        :param X_test: DataFrame containing test texts and metadata.
        :param y_test: List of test labels.
        :return: Classification report as a string.
        """
        y_pred = self.best_model.predict(X_test)
        report = classification_report(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')

        return {
            'classification_report': report,
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall
        }

    def predict(self, X):
        """
        Predict the labels for the provided data.

        :param X: DataFrame containing texts and metadata.
        :return: Predicted labels.
        """
        return self.best_model.predict(X)
