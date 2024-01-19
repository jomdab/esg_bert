import pandas as pd
import string
import re
import spacy
from sklearn.model_selection import train_test_split
import numpy as np
from nltk.corpus import wordnet
import random
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
# from googletrans import Translator
# nltk.download('wordnet')

def get_synonyms(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.append(lemma.name())
    return synonyms

# Function to perform synonym replacement
def synonym_replacement(text, n=2):
    words = text.split()
    new_words = words.copy()
    random_word_list = list(set([word for word in words ]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) > 0:
            synonym = random.choice(synonyms)
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break
    return ' '.join(new_words)

# Function to perform data augmentation
def augment_data(sentences, labels, augmentation_factor=1):
    augmented_sentences = []
    augmented_labels = []
    for sentence, label in zip(sentences, labels):
        augmented_sentences.append(sentence)
        augmented_labels.append(label)
        for _ in range(augmentation_factor):
            augmented_text = synonym_replacement(sentence)
            augmented_sentences.append(augmented_text)
            augmented_labels.append(label)
    return augmented_sentences, augmented_labels

def back_translate(sentence, translator_en_to_de, tokenizer, model_de_to_en):

    # Translate to the target language
    en_to_de_output = translator_en_to_de(sentence)
    translated_text = en_to_de_output[0]['translation_text']

    # Translate back to the source language
    input_ids = tokenizer(translated_text, return_tensors="pt", add_special_tokens=False).input_ids
    output_ids = model_de_to_en.generate(input_ids)[0]
    back_translation = tokenizer.decode(output_ids, skip_special_tokens=True)

    return back_translation

def augment_data_back_translation(sentences, labels, translator_en_to_de, tokenizer, model_de_to_en ,augmentation_factor=1 ):
    augmented_sentences = []
    augmented_labels = []
    c = 1
    for sentence, label in zip(sentences, labels):
        print(c)
        augmented_sentences.append(sentence)
        augmented_labels.append(label)

        # Apply back translation
        for _ in range(augmentation_factor):
            try:
                translated_sentence = back_translate(sentence, translator_en_to_de, tokenizer, model_de_to_en)
                augmented_sentences.append(translated_sentence)
                augmented_labels.append(label)
            except Exception as e:
                print(f"Error tokenizing sentence: {e}")
        c+=1
    return augmented_sentences,augmented_labels

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

print(sentences[979])
# 37
train_text, test_val_text, train_labels, test_val_labels =  train_test_split(sentences, labels, test_size=0.2, random_state=69)
test_text, val_text, test_labels, val_labels =  train_test_split(test_val_text, test_val_labels, test_size=0.5, random_state=42)

# Augment the training data
print("augment data back translation")
#English to German using the Pipeline and T5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
translator_en_to_de = pipeline("translation_en_to_de", model='t5-base', device=device)
#Germal to English using Bert2Bert model
tokenizer = AutoTokenizer.from_pretrained("google/bert2bert_L-24_wmt_de_en", pad_token="<pad>", eos_token="</s>", bos_token="<s>")
model_de_to_en = AutoModelForSeq2SeqLM.from_pretrained("google/bert2bert_L-24_wmt_de_en")

train_text, train_labels = augment_data_back_translation(train_text, labels,translator_en_to_de, tokenizer, model_de_to_en, augmentation_factor=1)

print("synonym replacement")
augmented_train_text, augmented_train_labels = augment_data(train_text, train_labels, augmentation_factor=1)

# Combine original and augmented data
train_text = np.concatenate([train_text, augmented_train_text])
train_labels = np.concatenate([train_labels, augmented_train_labels])

# Create a DataFrame and save to CSV
augmented_data = pd.DataFrame({'ESGN': train_text, 'class': train_labels})
augmented_data.to_csv('train.csv', index=False)
augmented_data = pd.DataFrame({'ESGN': test_text, 'class': test_labels})
augmented_data.to_csv('test.csv', index=False)
augmented_data = pd.DataFrame({'ESGN': val_text, 'class': val_labels})
augmented_data.to_csv('val.csv', index=False)


print(len(train_text))