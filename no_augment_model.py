import pandas as pd
import string
import re
import spacy
from transformers import BertTokenizer, BertForSequenceClassification, pipeline ,BertModel,DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import ElectraForSequenceClassification, ElectraTokenizer
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torch.optim import AdamW
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import Counter
import matplotlib.pyplot as plt
from nltk.corpus import wordnet
import random
import nltk
nltk.download('wordnet')

data = pd.read_csv("esgn_all.csv", encoding= 'unicode_escape')

# Define the mapping for label values
label_mapping = {0: 1, 1: 2, 2: 3, 3: 0}

# Apply the mapping to the 'label' column
data['class'] = data['class'].map(label_mapping)

# Removing Punctuations
data['ESGN'] = data['ESGN'].apply(lambda x: x.translate(str.maketrans('','', string.punctuation)))

# Removing urls
data['ESGN']=data['ESGN'].apply(lambda x : re.compile(r'https?://\S+|www\.\S+').sub('',x))

# Removing HTML Tags
data['ESGN']=data['ESGN'].apply(lambda x : re.compile(r'<.*?>').sub('',x))

# Removing emoji tags
emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
data['ESGN']=data['ESGN'].apply(lambda x : emoji_pattern.sub('',x))

# lowercase
data['ESGN']=data['ESGN'].apply(lambda x : x.lower())

# Stop word
nlp = spacy.load("en_core_web_sm")
def stop_word(text):
    temp=[]
    for t in nlp(text):
        if not nlp.vocab[t.text].is_stop :
            temp.append(t.text)
    return " ".join(temp)

data['ESGN']=data['ESGN'].apply(lambda x : stop_word(x) )

# Split data into train and validation sets
sentences = data['ESGN'].values
labels = data['class'].values

# 37
train_text, test_val_text, train_labels, test_val_labels =  train_test_split(sentences, labels, test_size=0.2, random_state=69)
test_text, val_text, test_labels, val_labels =  train_test_split(test_val_text, test_val_labels, test_size=0.5, random_state=42)

# load a pre-trained BERT tokenizer and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = ElectraTokenizer.from_pretrained("google/electra-small-discriminator")
bert_model = ElectraForSequenceClassification.from_pretrained("google/electra-small-discriminator", num_labels=4)
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)
# bert_model = BertModel.from_pretrained('model')
# tokenizer = DistilBertTokenizer.from_pretrained('bert-large-uncased')
# bert_model = DistilBertForSequenceClassification.from_pretrained('bert-large-uncased', num_labels=4)
bert_model = bert_model.to(device)
# nlp = pipeline("text-classification", model=bert_model, tokenizer=tokenizer)

# tokenize the text data
train_list = [str(t) for t in train_text]
val_list = [str(t) for t in val_text]
test_list = [str(t) for t in test_text]
train_inputs = tokenizer(train_list, padding=True, truncation=True, return_tensors="pt")
val_inputs = tokenizer(val_list, padding=True, truncation=True, return_tensors="pt")
test_inputs= tokenizer(test_list, padding=True, truncation=True, return_tensors="pt")

# convert labels to PyTorch tensors
train_labels = torch.tensor(train_labels, dtype=torch.float32)
val_labels = torch.tensor(val_labels, dtype=torch.float32)
test_labels = torch.tensor(test_labels, dtype=torch.float32)

# create DataLoader for training and validation sets
train_dataset = TensorDataset(train_inputs.input_ids, train_inputs.attention_mask, train_labels)
val_dataset = TensorDataset(val_inputs.input_ids, val_inputs.attention_mask, val_labels)
test_dataset = TensorDataset(test_inputs.input_ids, test_inputs.attention_mask, test_labels)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)
test_loader = DataLoader(test_dataset, batch_size=8)


# define optimizer and loss function
optimizer = AdamW(bert_model.parameters(), lr=1e-5)
loss_fn = nn.CrossEntropyLoss()
for param in bert_model.parameters():
    param = param.to(device)

# Set the number of epochs
num_epochs = 1

train_losses = []
val_losses = []
val_accuracys = []
train_accuracys = []
best_accuracy= 0

# Training loop
for epoch in range(num_epochs):
    bert_model.train()
    train_predictions = []
    train_true_labels = []
    train_loss = 0.0
    print(f"Epoch {epoch+1}/{num_epochs}")

    for batch in train_loader:
        input_ids, attention_mask, labels = batch

        # Move input data to GPU
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = bert_model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # Use torch.nn.CrossEntropyLoss, which combines log_softmax and NLLLoss
        labels = labels.to(torch.long)
        loss = torch.nn.functional.cross_entropy(logits, labels)

        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        # Apply softmax to obtain probabilities
        probabilities = torch.nn.functional.softmax(logits, dim=-1)

        # Get predicted class (index with the highest probability)
        predicted_class = torch.argmax(probabilities, dim=-1)

        train_predictions.extend(predicted_class.cpu().numpy())
        train_true_labels.extend(labels.cpu().numpy())

    train_accuracy = accuracy_score(train_true_labels, train_predictions)
    train_accuracys.append(train_accuracy*100)

    # Calculate average training loss for the epoch
    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    print(f"Train Loss: {avg_train_loss}")

    # Validation
    bert_model.eval()
    test_predictions = []
    test_true_labels = []
    val_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, labels = batch

            # Move input data to GPU
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = bert_model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            labels = labels.to(torch.long)
            loss = loss_fn(logits, labels)
            val_loss += loss.item()
            # Apply softmax to obtain probabilities
            probabilities = torch.nn.functional.softmax(logits, dim=-1)

            # Get predicted class (index with the highest probability)
            predicted_class = torch.argmax(probabilities, dim=-1)

            test_predictions.extend(predicted_class.cpu().numpy())
            test_true_labels.extend(labels.cpu().numpy())

    val_accuracy = accuracy_score(test_true_labels, test_predictions)
    val_accuracys.append(val_accuracy*100)

    # Calculate average validation loss for the epoch
    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    print(f"Validation Loss: {avg_val_loss}")
    print(f"Validation Accuracy: {val_accuracy*100}")

    # Save the model if validation loss is the best so far
    # if val_accuracy > best_accuracy:
    #     best_accuracy = val_accuracy
    #     best_epoch = epoch
    #     # Save the model
    #     model_save_path = f'model_best_val_loss'
    #     bert_model.save_pretrained(model_save_path)
    #     tokenizer.save_pretrained(model_save_path)

# for test set
test_loss=0
test_predictions = []
test_true_labels = []
with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask, labels = batch

        # Move input data to GPU
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        outputs = bert_model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        labels = labels.to(torch.long)
        loss = loss_fn(logits, labels)
        test_loss += loss.item()
        # Apply softmax to obtain probabilities
        probabilities = torch.nn.functional.softmax(logits, dim=-1)

        # Get predicted class (index with the highest probability)
        predicted_class = torch.argmax(probabilities, dim=-1)

        test_predictions.extend(predicted_class.cpu().numpy())
        test_true_labels.extend(labels.cpu().numpy())

test_accuracy = accuracy_score(test_true_labels, test_predictions)

# Calculate average validation loss for the epoch
avg_test_loss = test_loss / len(test_loader)
    

# Plotting the training and validation losses
plt.plot(range(1, num_epochs + 1), train_accuracys, label='Training accuracy')
plt.plot(range(1, num_epochs + 1), val_accuracys, label='Validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

# Calculate precision, recall, and F1 score
precision = precision_score(test_true_labels, test_predictions, average='weighted')
recall = recall_score(test_true_labels, test_predictions, average='weighted')
f1 = f1_score(test_true_labels, test_predictions, average='weighted')

print(f"Test Loss: {test_loss / len(val_loader)}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1 Score: {f1 * 100:.2f}%")


# model_save_path = 'google/electra_small_discriminator'
# bert_model.save_pretrained(model_save_path)
# tokenizer.save_pretrained(model_save_path)