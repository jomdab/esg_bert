import pandas as pd
import string
import re
import spacy
from transformers import BertTokenizer, BertForSequenceClassification, pipeline ,BertModel
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torch.optim import AdamW
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import Counter

data = pd.read_csv("esgn_all.csv", encoding= 'unicode_escape')

# Define the mapping for label values
label_mapping = {0: 1, 1: 2, 2: 3, 3: 0}

# Apply the mapping to the 'label' column
data['class'] = data['class'].map(label_mapping)

# Split data into train and validation sets
sentences = data['ESGN'].values
labels = data['class'].values
train_text, test_val_text, train_labels, test_val_labels =  train_test_split(sentences, labels, test_size=0.2, random_state=69)
# 0,21
test_text, val_text, test_labels, val_labels =  train_test_split(test_val_text, test_val_labels, test_size=0.5, random_state=0)

labels = torch.tensor(labels, dtype=torch.float32)
val_labels = torch.tensor(val_labels, dtype=torch.float32)
train_labels = torch.tensor(train_labels, dtype=torch.float32)
test_labels = torch.tensor(test_labels, dtype=torch.float32)

# print the number of occurrences for each class in all sentence
class_counts = Counter(labels.numpy())
print("Number of occurrences for each class in all labels:")
for class_label, count in class_counts.items():
    print(f"Class {class_label}: {count}")

    # print the number of occurrences for each class in val_labels
class_counts = Counter(train_labels.numpy())
print("Number of occurrences for each class in train_labels:")
for class_label, count in class_counts.items():
    print(f"Class {class_label}: {count}")

    class_counts = Counter(test_labels.numpy())
print("Number of occurrences for each class in test_labels:")
for class_label, count in class_counts.items():
    print(f"Class {class_label}: {count}")

# print the number of occurrences for each class in val_labels
class_counts = Counter(val_labels.numpy())
print("Number of occurrences for each class in val_labels:")
for class_label, count in class_counts.items():
    print(f"Class {class_label}: {count}")


