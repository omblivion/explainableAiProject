import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sklearn.model_selection import train_test_split
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
        # Map the sentiment label to the target value
        if label == "negative" or label == "Negative":
            return -1
        elif label == "neutral" or label == "Neutral":
            return 0
        elif label == "positive" or label == "Positive":
            return 1
        else:
            return None

    # Generate synthetic data using LLMs to be defined
    def generate_synthetic_data(self, topic, n_samples):
        openai.api_key = 'YOUR_API_KEY'
        synthetic_data = []
        for _ in range(n_samples):
            prompt = f"Generate six tweets related to {topic} that expresses sentiment."
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                max_tokens=60
            )
            synthetic_data.append(response.choices[0].text.strip())
        return synthetic_data

    def augment_training_data(self, topics, n_samples=100):
        augmented_data = {'text': [], 'label': []}
        augmented_data_with_topics = {'text': [], 'label': [], 'topic': []}
        for topic in topics:
            synthetic_texts = self.generate_synthetic_data(topic, n_samples)
            # Assuming the sentiment label for generated data
            augmented_data['text'].extend(synthetic_texts)
            augmented_data['label'].extend([1] * len(synthetic_texts))  # Defaulting to neutral
            augmented_data_with_topics['text'].extend(synthetic_texts)
            augmented_data_with_topics['label'].extend([1] * len(synthetic_texts))
            augmented_data_with_topics['topic'].extend([topic] * len(synthetic_texts))

        augmented_df = pd.DataFrame(augmented_data)
        augmented_df_with_topics = pd.DaataFrame(augmented_data_with_topics)
        return augmented_df, augmented_df_with_topics

    def fine_tune_with_augmented_data(self, topics, n_samples=100, epochs=3, batch_size=16, learning_rate=2e-5):
        augmented_train_data, augmented_train_data_with_topics = self.augment_training_data(topics, n_samples)
        return self.fine_tune(augmented_train_data, epochs, batch_size, learning_rate), augmented_train_data_with_topics

    # Fine-tune the model on a custom dataset
    def fine_tune(self, df, epochs=3, batch_size=16, learning_rate=2e-5):
        # Preprocess the dataset
        df = df.rename(columns={"clean_text": "text", "category": "label"})     # Rename the columns
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

        # Define training arguments
        training_args = TrainingArguments(  # Define the training arguments
            output_dir="./results",
            evaluation_strategy="epoch",
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
        )

        # Fine-tune the model
        trainer.train()

        # Evaluate the model
        results = trainer.evaluate()
        print(results)
        return results
