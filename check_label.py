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
train_text, val_text, train_labels, val_labels =  train_test_split(sentences, labels, test_size=0.1, random_state=69)

val_labels = torch.tensor(val_labels, dtype=torch.float32)
train_labels = torch.tensor(train_labels, dtype=torch.float32)

# print the number of occurrences for each class in val_labels
class_counts = Counter(val_labels.numpy())
print("Number of occurrences for each class in val_labels:")
for class_label, count in class_counts.items():
    print(f"Class {class_label}: {count}")

    # print the number of occurrences for each class in val_labels
class_counts = Counter(train_labels.numpy())
print("Number of occurrences for each class in train_labels:")
for class_label, count in class_counts.items():
    print(f"Class {class_label}: {count}")
