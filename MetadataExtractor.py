import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

class MetadataExtractor:
    def __init__(self):
        # Check if GPUs are available and set the devices accordingly
        self.devices = [i for i in range(torch.cuda.device_count())]

        # Initialize the zero-shot classification pipelines with specific models
        self.MODEL = "roberta-large-mnli"
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL)
        self.models = [
            AutoModelForSequenceClassification.from_pretrained(self.MODEL, ignore_mismatched_sizes=True).to(f'cuda:{device}')
            for device in self.devices
        ]
        self.classifiers = [
            pipeline("zero-shot-classification", model=model, tokenizer=self.tokenizer, device=device)
            for model, device in zip(self.models, self.devices)
        ]
        self.current_device_index = 0

    def _get_next_classifier(self):
        """
        Get the next classifier in a round-robin manner to distribute the workload.
        """
        classifier = self.classifiers[self.current_device_index]
        self.current_device_index = (self.current_device_index + 1) % len(self.devices)
        return classifier

    def extract_attribute(self, text, candidate_labels, hypothesis_template):
        """
        Extracts an attribute from the given text using the zero-shot classification model.

        :param text: The text to classify.
        :param candidate_labels: A list of strings representing candidate labels.
        :param hypothesis_template: A template for the hypothesis.
        :return: The label with the highest probability.
        """
        # Get the classifier for the current task
        classifier = self._get_next_classifier()
        # Perform zero-shot classification
        result = classifier(text, candidate_labels, hypothesis_template=hypothesis_template)
        # Get the label with the highest probability
        top_label = result['labels'][0]
        return top_label

    def extract_gender(self, text, candidate_labels):
        """
        Extracts the gender from the given text.

        :param candidate_labels:
        :param text: The text to classify.
        :return: The gender label with the highest probability.
        """
        hypothesis_template = "The person described in this text is {}."
        return self.extract_attribute(text, candidate_labels, hypothesis_template)

    def extract_topic(self, text, candidate_labels):
        """
        Extracts the topic from the given text.

        :param candidate_labels:
        :param text: The text to classify.
        :return: The topic label with the highest probability.
        """
        hypothesis_template = "The topic of this text is {}."
        return self.extract_attribute(text, candidate_labels, hypothesis_template)
