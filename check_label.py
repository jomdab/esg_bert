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
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

data = pd.read_excel("dataset.xlsx")



# Split data into train and validation sets
sentences = data['ESGN'].values
labels = data['class'].values
ori_text = sentences

# Use stratified sampling to split the data
stratified_splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=69)
train_indices, val_indices = next(stratified_splitter.split(sentences, labels, ori_text))

# Split the data based on the indices obtained from stratified sampling
train_text = [sentences[i] for i in train_indices]
val_text = [sentences[i] for i in val_indices]
train_labels = [labels[i] for i in train_indices]
val_labels = [labels[i] for i in val_indices]
train_ori_text = [ori_text[i] for i in train_indices]
val_ori_text = [ori_text[i] for i in val_indices]

labels = torch.tensor(labels, dtype=torch.float32)
val_labels = torch.tensor(val_labels, dtype=torch.float32)
train_labels = torch.tensor(train_labels, dtype=torch.float32)


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

# print the number of occurrences for each class in val_labels
class_counts = Counter(val_labels.numpy())
print("Number of occurrences for each class in val_labels:")
for class_label, count in class_counts.items():
    print(f"Class {class_label}: {count}")



