import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline


class MetadataExtractor:
    def __init__(self):
        # Check if GPU is available and set the device accordingly
        self.device = 0 if torch.cuda.is_available() else -1
        # Initialize the zero-shot classification pipeline with a specific model
        self.MODEL = "roberta-large-mnli"
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.MODEL).to(self.device)
        self.classifier = pipeline("zero-shot-classification", model=self.model, tokenizer=self.tokenizer,
                                   device=self.device)

    def extract_attribute(self, text, candidate_labels, hypothesis_template):
        """
        Extracts an attribute from the given text using the zero-shot classification model.

        :param text: The text to classify.
        :param candidate_labels: A list of strings representing candidate labels.
        :param hypothesis_template: A template for the hypothesis.
        :return: The label with the highest probability.
        """
        # Perform zero-shot classification
        result = self.classifier(text, candidate_labels, hypothesis_template=hypothesis_template)
        # Get the label with the highest probability
        top_label = result['labels'][0]
        return top_label

    def extract_gender(self, text):
        """
        Extracts the gender from the given text.

        :param text: The text to classify.
        :return: The gender label with the highest probability.
        """
        candidate_labels = ["male", "female", "unknown"]
        hypothesis_template = "The person described in this text is {}."
        return self.extract_attribute(text, candidate_labels, hypothesis_template)

    def extract_race(self, text):
        """
        Extracts the race from the given text.

        :param text: The text to classify.
        :return: The race label with the highest probability.
        """
        candidate_labels = ["white", "black", "asian", "hispanic", "other", "non-identified"]
        hypothesis_template = "The person described in this text is {}."
        return self.extract_attribute(text, candidate_labels, hypothesis_template)
