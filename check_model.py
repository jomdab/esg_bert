import torch
from torchsummary import summary
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from esg_bert import BertClassifier  

# Define model name and number of classes
model_name = "roberta-base"
num_classes = 8  # Adjust according to your task

# Load pre-trained RoBERTa model and tokenizer
tokenizer = RobertaTokenizer.from_pretrained(model_name)
bert_model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)

# Instantiate the BertClassifier model
classifier = BertClassifier(bert_model, num_classes)

# Choose an appropriate batch size and max sequence length for your dataset
batch_size = 8
max_seq_length = 128

# Print the model summary
summary(classifier, input_size=[(batch_size, max_seq_length), (batch_size, max_seq_length)])
