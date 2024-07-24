import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline


class SentimentAnalyzer:
    def __init__(self):
        self.model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        self.device = 0 if torch.cuda.is_available() else -1  # Use GPU if available
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self.device)
        self.classifier = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer, device=self.device)

    def analyze_sentiment(self, text):
        results = self.classifier(text)
        return results[0]['label']

    def map_label_to_target(self, label):
        # Map the sentiment label to the target value
        if label == "negative":
            return 0
        elif label == "neutral":
            return 2
        elif label == "positive":
            return 4
        else:
            return None
