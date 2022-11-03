from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from datasets import DatasetDict
import torch
import json
from datasets import load_metric
import numpy as np

class PANCDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class PANCDatasetTest(PANCDataset):
    def __init__(self, encodings, labels):
        super().__init__(encodings, labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item


class ESPD:
    '''
    wannabe Implementation of the Early Sexual Predator Detection paper
    '''

    def __init__(self):
        print("loading tokenizer and model...")
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

    def tokenize_function(self, examples):
        return self.tokenizer(examples, padding="max_length", truncation=True)

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return self.metric.compute(predictions=predictions, references=labels)

    def create_sliding_windows(self, segment, window_size=50):
        windows = []
        seg_len = len(segment)
        #print("segment:", segment)
        if window_size > seg_len:
            windows.append(" ".join(segment))
        else:
            for i in range(0, seg_len-window_size+1):
                #print("single window:", segment[i:i+window_size])
                windows.append(" ".join(segment[i:i+window_size]))
        #print("windows:", windows)
        return windows, len(windows)

    def get_preprocessed_data(self, data):
        data_temp = data.to_numpy()[:,-2:] # format: first column: chat segments; second column: numerical label
        segments, labels = data_temp[:,0], data_temp[:,1]
        windows = []
        labels_windows = []
        for i, segment in enumerate(segments):
            windows_curr_segment, n_windows = self.create_sliding_windows(segment)
            #print(len(windows_curr_segment))
            windows += windows_curr_segment
            labels_windows += [labels[i]] * len(windows_curr_segment)
            #if i > 0:
            #    break
        return windows, labels_windows

    def preprocess(self, data):
        '''
        Input X_train and X_test of dataset(should be pd.Dataframe)
        returns preprocessed Huggingface dataset
        '''

        #print(X_train.loc[:, 'segment'])
        data.loc[:, 'segment'] = data['segment'].apply(lambda x: json.loads(x)['messages'])

        data.loc[:, 'label_numerical'] = (data['label'] == 'predator').apply(int)

        print("preprocessing data...")
        texts, labels = self.get_preprocessed_data(data)

        print("tokenizing...")
        encodings = self.tokenize_function(texts)

        dataset = PANCDataset(encodings, labels)
        #dataset = DatasetDict()
        #dataset['train'] = train_dataset
        #dataset['test'] = test_dataset
        return dataset

    def fit(self, dataset):
        # Assume data is preprocessed
        #tokenized_datasets = dataset.map(self.tokenize_function, batched=True)
        #small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
        #small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["eval"],
            compute_metrics=self.compute_metrics,
        )

        trainer.train()
        trainer.save_model()
