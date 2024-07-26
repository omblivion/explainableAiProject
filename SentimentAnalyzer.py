import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Suppress specific warnings
warnings.filterwarnings("ignore", message="Some weights of the model checkpoint.*")
class SentimentAnalyzer:
    def __init__(self):
        self.model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        self.device = 0 if torch.cuda.is_available() else -1  # Use GPU if available
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, ignore_mismatched_sizes=True).to(self.device)
        self.classifier = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer, device=self.device)

    def analyze_sentiment(self, text):
        results = self.classifier(text)
        return results[0]['label']

    def map_label_to_target(self, label):
        # Map the sentiment label to the target value
        if label == "negative" or label == "Negative":
            return 0
        elif label == "neutral" or label == "Neutral":
            return 2
        elif label == "positive" or label == "Positive":
            return 4
        else:
            return None
