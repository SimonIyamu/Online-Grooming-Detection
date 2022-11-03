import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from xgboost import plot_importance
from matplotlib import pyplot

train_dataset = pd.read_csv("../../Datasets/PANC/additional_features/PANC-train_add_features_emotion.csv", index_col=0)
test_dataset = pd.read_csv("../../Datasets/PANC/additional_features/PANC-test_add_features_emotion.csv", index_col=0)

data_classes = ["predator","non-predator"]
y_train = train_dataset['label'].apply(data_classes.index)
X_train = train_dataset.drop(['chatName','segment', 'label'], axis=1)

y_test = test_dataset['label'].apply(data_classes.index)
X_test = test_dataset.drop(['chatName','segment', 'label'], axis=1)

#X_train, X_test, y_train, y_test = train_test_split(X, y)

# XGboost and feature selection

model = XGBClassifier()
model.fit(X_train, y_train)

# Plot imporance
#plot_importance(model)
#pyplot.show()

# make predictions for test data
y_pred1 = model.predict_proba(X_test)[:,1]

y_pred2 = pd.read_csv('y_pred_BERT.csv')['LABEL_1']

# Aggregate predictions
weight_1 = 0.5
weight_2 = 0.5
#y_pred = np.maximum(y_pred1,y_pred2)
#y_pred = np.minimum(y_pred1,y_pred2)
y_pred = np.average(np.vstack((y_pred1, y_pred2)), axis=0, weights=[weight_1, weight_2])

#y_pred = y_pred2


predictions = [round(value) for value in y_pred]

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average='macro')
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
print("Accuracy: %.3f%%" % (accuracy * 100.0))
print("F1: %.3f" % (f1))
print("Precision: %.3f%%" % (precision * 100.0))
print("Recall: %.3f%%" % (recall * 100.0))