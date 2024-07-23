from transformers import pipeline

class MetadataExtractor:
    def __init__(self):
        # Initialize the zero-shot classification pipeline with a specific model
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

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