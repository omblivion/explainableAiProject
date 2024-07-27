import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, \
    DataCollatorWithPadding, pipeline, AutoModelForSeq2SeqLM
from sklearn.preprocessing import OneHotEncoder
from datasets import Dataset


class SentimentAnalyzer:
    def __init__(self):
        self.model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        self.device = 0 if torch.cuda.is_available() else -1  # Use GPU if available
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, ignore_mismatched_sizes=True).to(self.device)
        self.classifier = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer, device=self.device)

        # Initialize FLAN model for synthetic data generation
        self.flan_model_name = "google/flan-t5-small"
        self.flan_tokenizer = AutoTokenizer.from_pretrained(self.flan_model_name)
        self.flan_model = AutoModelForSeq2SeqLM.from_pretrained(self.flan_model_name).to(self.device)
    def analyze_sentiment(self, text):
        results = self.classifier(text)
        return results[0]['label']

    def map_label_to_target(self, label):
        # Map the sentiment label to the target value
        if label == "negative" or label == "Negative":
            return 0
        elif label == "neutral" or label == "Neutral":
            return 1
        elif label == "positive" or label == "Positive":
            return 2
        else:
            return None

    def map_target_to_label(self, target):
        # Map the target value to the sentiment label
        if target == 0:
            return "negative"
        elif target == 1:
            return "neutral"
        elif target == 2:
            return "positive"
        else:
            return None

    # Generate synthetic data using the FLAN model
    def generate_synthetic_data(self, topic, text, sentiment, n_samples):
        synthetic_data = []
        #print(f"Generating synthetic data for topic: {topic}, text: {text}, sentiment: {sentiment}")
        for _ in range(n_samples):
            prompt = f"Generate a tweet related to {topic} that expresses a {sentiment} sentiment and the tweet has to be semantically similar to: '{text}' "
            inputs = self.flan_tokenizer(prompt, return_tensors="pt").to(self.device)
            # Use top-k sampling and temperature sampling for more diverse outputs
            outputs = self.flan_model.generate(
                inputs.input_ids,
                max_length=60,
                num_return_sequences=1,
                do_sample=True,
                top_k=50,  # Consider top 50 tokens
                temperature=0.5  # Adjust temperature to control diversity
            )
            generated_text = self.flan_tokenizer.decode(outputs[0], skip_special_tokens=True)
            synthetic_data.append(generated_text)
            #print(f"Generated Text: {generated_text}")
        return synthetic_data

    # Augment the training data with synthetic data
    def generate_training_data(self, topics, texts, sentiments, n_samples=6):
        generated_data = {'text': [], 'category': []}
        generated_data_with_topic = {'text': [], 'category': [], 'topic': []}

        for topic, text, sentiment in zip(topics, texts, sentiments):
            sentiment_text = self.map_target_to_label(sentiment)
            synthetic_texts = self.generate_synthetic_data(topic, text, sentiment_text, n_samples)   # List of synthetic texts
            generated_data['text'].extend(synthetic_texts)
            generated_data['category'].extend([sentiment] * len(synthetic_texts))    # append sentiment to texts many times
            generated_data_with_topic['text'].extend(synthetic_texts)
            generated_data_with_topic['category'].extend([sentiment] * len(synthetic_texts))
            generated_data_with_topic['topic'].extend([topic] * len(synthetic_texts))

        generated_df = pd.DataFrame(generated_data)
        generated_df_with_topics = pd.DataFrame(generated_data_with_topic)
        return generated_df, generated_df_with_topics


    # Fine-tune the model on a custom dataset
    def fine_tune(self, df, epochs=3, batch_size=16, learning_rate=2e-5):
        # Preprocess the dataset
        df = df.rename(columns={"text": "text", "category": "label"})     # Rename the columns
        df['label'] = df['label'].astype(int)   # Ensure the labels are integers
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)        # Split the dataset

        train_dataset = Dataset.from_pandas(train_df)   # Load the dataset
        test_dataset = Dataset.from_pandas(test_df)

        def tokenize_function(examples):    # Tokenize the text
            return self.tokenizer(examples["text"], padding="max_length", truncation=True)

        train_dataset = train_dataset.map(tokenize_function, batched=True)  # Tokenize the dataset
        test_dataset = test_dataset.map(tokenize_function, batched=True)

        train_dataset = train_dataset.remove_columns(["text"])  # Remove the text column after tokenization
        test_dataset = test_dataset.remove_columns(["text"])

        train_dataset.set_format("torch")   # Set the format to PyTorch
        test_dataset.set_format("torch")

        # Define the data collator
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        # Define training arguments
        training_args = TrainingArguments(  # Define the training arguments
            output_dir="./results",
            run_name="finetuning_sentiment_classifier",
            eval_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
        )

        # Define the trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            data_collator=data_collator,
        )

        # Fine-tune the model
        trainer.train()

        # Evaluate the model
        results = trainer.evaluate()
        print(results)
        return results
