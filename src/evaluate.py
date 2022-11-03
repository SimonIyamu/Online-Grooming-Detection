import argparse
import json
import os
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoConfig, TrainingArguments, Trainer, BertModel, AutoModelForSequenceClassification
from datasets import load_metric, load_dataset

# two arguments needed for this script:
# 1) path to folder of model checkpoint
# 2) path to file with test data

parser = argparse.ArgumentParser(description='Evaluate supplied model on the test set')
parser.add_argument(
	"--model",
	dest='model',
	help="path to the model checkpoint",
	required=True
)
parser.add_argument(
	"--test_file",
	dest='test_file',
	help="path to the file which contains the test data",
    required=True
)

def load_model_and_tokenizer(model_path):
    # load tokenizer
    #print(f"loading tokenizer from {path_to_model}...")
    #tokenizer = AutoTokenizer.from_pretrained(model_path, config=AutoConfig.from_pretrained(path_to_model))
    #_name_or_path
    config_path = os.path.join(path_to_model, "config.json")
    with open(config_path, 'r') as f:
        config_json = json.load(f)
    print("loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config_json["_name_or_path"])
    print(f"loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(model_path, config=AutoConfig.from_pretrained(path_to_model))
    return model, tokenizer, config_json

args = parser.parse_args()
path_to_model = args.model
path_to_test_file = args.test_file
model, tokenizer, model_config = load_model_and_tokenizer(path_to_model)

train_args = TrainingArguments(
    output_dir='./predictions'
)

accuracy = load_metric('accuracy')
f1 = load_metric('f1')
precision = load_metric('precision')
recall = load_metric('recall')


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy.compute(predictions=predictions, references=labels)
    f1_score = f1.compute(predictions=predictions, references=labels)
    prec = precision.compute(predictions=predictions, references=labels)
    rec = recall.compute(predictions=predictions, references=labels)
    return {'accuracy': acc, 'f1': f1_score, 'precision': prec, 'recall': rec}


def tokenize_function(batch):
        '''
        1)Tokenize segments
        2)change labels to numerical
        '''
        tokenized_batch = tokenizer(batch["segment"], padding="max_length", truncation=True)
        tokenized_batch["label"] = [int(label == "predator") for label in batch["label"]]
        return tokenized_batch

trainer = Trainer(
    model=model,
    args=train_args,
    compute_metrics=compute_metrics,
    #eval_dataset=test_data,
    tokenizer=tokenizer
)

test_dataset = load_dataset('csv', data_files=path_to_test_file)
print("test_dataset:", test_dataset)
tokenized_dataset = test_dataset.map(tokenize_function, batched=True)
print("tokenized_dataset:", tokenized_dataset)
test_dataset = tokenized_dataset["train"].shuffle(seed=42)

logits, labels, metrics = trainer.predict(test_dataset)
predictions = np.argmax(logits)

print(metrics)

# can maybe be used for f1_latency evaluation
idx2chatID = dict()
for i, sample in enumerate(test_dataset):
    idx2chatID[i] = test_dataset[i]['chatName']

