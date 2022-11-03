from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from datasets import load_metric
import numpy as np


class ESPD:
    '''
    wannabe Implementation of the Early Sexual Predator Detection paper
    '''

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        self.model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)

        self.training_args = TrainingArguments(
            output_dir='./results',          # output directory
            num_train_epochs=3,              # total number of training epochs
            per_device_train_batch_size=4,  # batch size per device during training
            gradient_accumulation_steps=4,
            per_device_eval_batch_size=16,   # batch size for evaluation
            warmup_steps=500,                # number of warmup steps for learning rate scheduler
            weight_decay=0.01,               # strength of weight decay
            logging_dir='./logs',            # directory for storing logs
            logging_steps=10,
            report_to="none",
            evaluation_strategy="epoch")
        self.metric = load_metric("f1")

    def tokenize_function(self, batch):
        '''
        1)Tokenize segments
        2)change labels to numerical
        '''
        tokenized_batch = self.tokenizer(batch["segment"], padding="max_length", truncation=True)
        tokenized_batch["label"] = [int(label == "predator") for label in batch["label"]]
        return tokenized_batch

    #def label_to_int(self, examples):
    #    return int(examples["label"] == "predator")

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return self.metric.compute(predictions=predictions, references=labels)

    def fit(self, dataset):
        tokenized_datasets = dataset.map(self.tokenize_function, batched=True)
        #tokenized_datasets = tokenized_datasets.map(self.label_to_int, batched=True)

        train_dataset = tokenized_datasets["train"].shuffle(seed=42)
        eval_dataset = tokenized_datasets["test"].shuffle(seed=42)

        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics,
        )

        trainer.train()
        trainer.save_model()
