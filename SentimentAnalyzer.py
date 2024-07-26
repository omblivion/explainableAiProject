import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, pipeline

from datasets import Dataset


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
        # Ensure the training data includes labels
        if 'label' not in train_data.columns:
            train_data['label'] = train_data['text'].apply(self.map_label_to_target)

        # Tokenize the data
        def tokenize_function(examples):
            return self.tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)

        train_dataset = Dataset.from_pandas(train_data)
        tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)

        # Remove columns not required for training
        tokenized_train_dataset = tokenized_train_dataset.remove_columns(["text", "__index_level_0__"])

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
            compute_metrics=self.compute_metrics
        )

        # Train the model
        trainer.train()

        return trainer.evaluate()

    def compute_metrics(self, p):
        preds = np.argmax(p.predictions, axis=1)
        accuracy = (preds == p.label_ids).astype(np.float32).mean().item()
        return {"accuracy": accuracy}
