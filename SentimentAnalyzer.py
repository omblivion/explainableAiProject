import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, pipeline


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
        if label == "negative" or label == "Negative":
            return -1
        elif label == "neutral" or label == "Neutral":
            return 0
        elif label == "positive" or label == "Positive":
            return 1
        else:
            return None

    def fine_tune(self, train_data):
        # Tokenize the data
        def tokenize_function(examples):
            return self.tokenizer(examples['text'], padding="max_length", truncation=True)

        train_dataset = Dataset.from_pandas(train_data)
        tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)

        # Set up training arguments
        training_args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=3,
            weight_decay=0.01,
        )

        # Initialize Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train_dataset,
            tokenizer=self.tokenizer,
        )

        # Train the model
        trainer.train()

        return trainer.evaluate()
