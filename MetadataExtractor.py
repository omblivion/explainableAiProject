import torch
from transformers import pipeline

class MetadataExtractor:
    def __init__(self):
        # Check if GPU is available and set the device accordingly
        device = 0 if torch.cuda.is_available() else -1
        # Initialize the zero-shot classification pipeline with a specific model
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device)

    def extract_probabilities(self, text, candidate_labels):
        """
        Extracts the probabilities for each candidate label from the given text using zero-shot classification.

        :param text: The text to classify.
        :param candidate_labels: A list of strings representing candidate topics.
        :return: A dictionary with candidate labels as keys and their corresponding probabilities as values.
        """
        result = self.classifier(text, candidate_labels, multi_label=True)
        probabilities = {label: result["scores"][i] for i, label in enumerate(result["labels"])}
        return probabilities
