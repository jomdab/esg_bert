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
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


data = pd.read_excel('gold_standard_corpus.xlsx')
data = data.dropna(subset=['headline'])


# Define the mapping for label values
label_mapping = {"environmental": 1, "social": 2, "governance": 3, "non-esg": 0}

# Apply the mapping to the 'label' column
data['class'] = data['esg_category'].map(label_mapping)

# Separate the dataset based on classes
class_0_data = data[data['class'] == 0]
class_1_data = data[data['class'] == 1]
class_2_data = data[data['class'] == 2]
class_3_data = data[data['class'] == 3]

# Randomly select 500 sentences for each class
test_set_class_0 = class_0_data.sample(n=500, random_state=42)
test_set_class_1 = class_1_data.sample(n=500, random_state=42)
test_set_class_2 = class_2_data.sample(n=500, random_state=42)
test_set_class_3 = class_3_data.sample(n=500, random_state=42)

# Combine the selected sentences to create the final test set
final_test_set = pd.concat([test_set_class_0, test_set_class_1, test_set_class_2, test_set_class_3])

# Shuffle the final test set
data = final_test_set.sample(frac=1, random_state=42).reset_index(drop=True)


# Removing Punctuations
data['headline'] = data['headline'].apply(lambda x: x.translate(str.maketrans('','', string.punctuation)))

# Removing urls
data['headline']=data['headline'].apply(lambda x : re.compile(r'https?://\S+|www\.\S+').sub('',x))

# Removing HTML Tags
data['headline']=data['headline'].apply(lambda x : re.compile(r'<.*?>').sub('',x))

# Removing emoji tags
emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
data['headline']=data['headline'].apply(lambda x : emoji_pattern.sub('',x))

# lowercase
data['headline']=data['headline'].apply(lambda x : x.lower())

# Stop word
nlp = spacy.load("en_core_web_sm")
def stop_word(text):
    temp=[]
    for t in nlp(text):
        if not nlp.vocab[t.text].is_stop :
            temp.append(t.text)
    return " ".join(temp)

data['headline']=data['headline'].apply(lambda x : stop_word(x) )

# Extract sentences and classes into variables
val_text = data['headline'].tolist()
val_labels = data['class'].tolist()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

paths = ['bert_base_uncased2']
name = ["bert"]

for i,path in enumerate(paths):
    # load a pre-trained BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(path)
    bert_model = BertForSequenceClassification.from_pretrained(path, num_labels=4)
    # bert_model = BertModel.from_pretrained('D:\esg_bert\model')
    bert_model = bert_model.to(device)
    # nlp = pipeline("text-classification", model=bert_model, tokenizer=tokenizer)

    # tokenize the text data
    val_list = [str(t) for t in val_text]
    val_inputs = tokenizer(val_list, padding=True, truncation=True, return_tensors="pt")

    # convert labels to PyTorch tensors
    val_labels = torch.tensor(val_labels, dtype=torch.float32)

    # create DataLoader for training and validation sets
    val_dataset = TensorDataset(val_inputs.input_ids, val_inputs.attention_mask, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=4)

    # define optimizer and loss function
    optimizer = AdamW(bert_model.parameters(), lr=1e-5)
    loss_fn = nn.CrossEntropyLoss()
    for param in bert_model.parameters():
        param = param.to(device)

    bert_model.eval()
    test_loss = 0.0
    test_predictions = []
    test_true_labels = []
    index = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, labels = batch

            # Move input data to GPU
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            # labels = labels.unsqueeze(1)

            text = val_text[index]
            outputs = bert_model(input_ids, attention_mask=attention_mask)

            # Correctly access logits based on the model output structure
            logits = outputs.logits if hasattr(outputs, "logits") else outputs.last_hidden_state

            # Use torch.nn.CrossEntropyLoss for multi-class classification
            labels = labels.to(torch.long)
            loss = loss_fn(logits, labels)
            test_loss += loss.item()

            # Apply softmax to obtain probabilities
            probabilities = torch.nn.functional.softmax(logits, dim=-1)

            # Get predicted class (index with the highest probability)
            predicted_class = torch.argmax(probabilities, dim=-1)

            test_predictions.extend(predicted_class.cpu().numpy())
            test_true_labels.extend(labels.cpu().numpy())
            # results = nlp(text)[0]["label"]
            # print(f"{index}:{predicted_class.numpy()} -> {results} , {labels.numpy()}")
            index += 1

    # calculate test accuracy
    test_accuracy = accuracy_score(test_true_labels, test_predictions)

    # Calculate precision, recall, and F1 score
    precision = precision_score(test_true_labels, test_predictions, average='weighted')
    recall = recall_score(test_true_labels, test_predictions, average='weighted')
    f1 = f1_score(test_true_labels, test_predictions, average='weighted')

    print(name[i])
    print(f"Test Loss: {test_loss / len(val_loader)}")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1 Score: {f1 * 100:.2f}%")

        # Compute the confusion matrix
    conf_matrix = confusion_matrix(test_true_labels, test_predictions)

    # Plot the confusion matrix
    class_names = ['environmental', 'environmental', 'social', 'non-esg']
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


