import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout , SimpleRNN
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import spacy
import re
import string
import seaborn as sns
from keras import Model
from keras import backend as K
from keras import initializers, regularizers, constraints
from keras.layers import Layer
import numpy as np

class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            # 1
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
            # next add a Dense layer (for classification/regression) or whatever...
            # 2
            hidden = LSTM(64, return_sequences=True)(words)
            sentence = Attention()(hidden)
            # next add a Dense layer (for classification/regression) or whatever...
        """
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0

        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(name='{}_W'.format(self.name),
                                 shape=(int(input_shape[-1]),),
                                 initializer=self.init,
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight(name='{}_b'.format(self.name),
                                     shape=(input_shape[1],),
                                     initializer='zero',
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        e = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))  # e = K.dot(x, self.W)
        if self.bias:
            e += self.b
        e = K.tanh(e)

        a = K.exp(e)
        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())
        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)

        c = K.sum(a * x, axis=1)
        return c

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.features_dim


class TextAttBiRNN(Model):
    def __init__(self,
                 maxlen,
                 max_features,
                 embedding_dims,
                 class_num=4,
#                  class_num=1, #old
#                  last_activation='sigmoid'): #old
                 last_activation='softmax'):
        super(TextAttBiRNN, self).__init__()
        self.maxlen = maxlen
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.class_num = class_num
        self.last_activation = last_activation
        self.embedding = Embedding(self.max_features, self.embedding_dims, input_length=self.maxlen)
        # self.bi_rnn = Bidirectional(LSTM(128, return_sequences=True))  # LSTM or GRU
        self.bi_rnn = Bidirectional(SimpleRNN(128, return_sequences=True))
        self.attention = Attention(self.maxlen)
        self.dropout = Dropout(0.2)
        self.classifier = Dense(self.class_num, activation=self.last_activation)

    def call(self, inputs):
        if len(inputs.get_shape()) != 2:
            raise ValueError('The rank of inputs of TextAttBiRNN must be 2, but now is %d' % len(inputs.get_shape()))
        if inputs.get_shape()[1] != self.maxlen:
            raise ValueError('The maxlen of inputs of TextAttBiRNN must be %d, but now is %d' % (self.maxlen, inputs.get_shape()[1]))
        embedding = self.embedding(inputs)
        x = self.bi_rnn(embedding)
        x = self.attention(x)
        output = self.classifier(x)
        return output
    


data = pd.read_excel("dataset.xlsx")

# Define the mapping for label values
label_mapping1 = {0: 1, 1: 0, 2: 0, 3: 1, 4: 1, 5: 0,6: 1,7: 0}
label_mapping2 = {0: 0, 1: 1, 2: 0, 3: 1, 4: 0, 5: 1,6: 1,7: 0}
label_mapping3 = {0: 0, 1: 0, 2: 1, 3: 0, 4: 1, 5: 1,6: 1,7: 0}

ori_text = data['ESGN'].values

# Apply the mapping to the 'label' column
labels = data['class'].values
labels1 = data['class'].map(label_mapping1).values
labels2 = data['class'].map(label_mapping2).values
labels3 = data['class'].map(label_mapping3).values

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

# 37
train_text, val_text, train_labels1, val_labels1,train_labels2, val_labels2,train_labels3, val_labels3,train_ori_text,val_ori_text,train_labels,val_labels =  train_test_split(sentences, labels1,labels2,labels3,ori_text,labels, test_size=0.1, random_state=1)

# Tokenize and pad sequences
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(train_text)

max_length = 100  # Adjust as needed
train_sequences = tokenizer.texts_to_sequences(train_text)
val_sequences = tokenizer.texts_to_sequences(val_text)


train_padded = tf.keras.preprocessing.sequence.pad_sequences(train_sequences, maxlen=max_length, padding='post')
val_padded = tf.keras.preprocessing.sequence.pad_sequences(val_sequences, maxlen=max_length, padding='post')


# # One-hot encode labels
# num_classes = len(set(train_labels1))
# train_labels_one_hot1 = tf.keras.utils.to_categorical(train_labels1, num_classes=num_classes)
# val_labels_one_hot1 = tf.keras.utils.to_categorical(val_labels1, num_classes=num_classes)
# num_classes = len(set(train_labels2))
# train_labels_one_hot2 = tf.keras.utils.to_categorical(train_labels2, num_classes=num_classes)
# val_labels_one_hot2 = tf.keras.utils.to_categorical(val_labels2, num_classes=num_classes)
# num_classes = len(set(train_labels3))
# train_labels_one_hot3 = tf.keras.utils.to_categorical(train_labels3, num_classes=num_classes)
# val_labels_one_hot3 = tf.keras.utils.to_categorical(val_labels3, num_classes=num_classes)


# Define the LSTM model
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 30
hidden_dim = 64

# Assuming your input tensor has shape (batch_size, sequence_length)
# Adjust parameters accordingly based on your actual data
model1 = TextAttBiRNN(max_features=150000, embedding_dims=50, maxlen=100, class_num=1, last_activation='sigmoid')
model2 = TextAttBiRNN(max_features=150000, embedding_dims=50, maxlen=100, class_num=1, last_activation='sigmoid')
model3 = TextAttBiRNN(max_features=150000, embedding_dims=50, maxlen=100, class_num=1, last_activation='sigmoid')

# Define optimizer and compile the model
learning_rate = 1e-3  
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model1.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
model2.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
model3.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping based on validation accuracy
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

# Model checkpoint to save the model when early stopping occurs
model_checkpoint1 = tf.keras.callbacks.ModelCheckpoint('rnn1', save_best_only=True)
model_checkpoint2 = tf.keras.callbacks.ModelCheckpoint('rnn2', save_best_only=True)
model_checkpoint3 = tf.keras.callbacks.ModelCheckpoint('rnn3', save_best_only=True)

# Train the model
num_epochs = 100
batch_size = 8
history1 = model1.fit(train_padded, train_labels1, epochs=num_epochs, batch_size=batch_size,
                    validation_data=(val_padded, val_labels1), callbacks=[early_stopping, model_checkpoint1])
history2 = model2.fit(train_padded, train_labels2, epochs=num_epochs, batch_size=batch_size,
                    validation_data=(val_padded, val_labels2), callbacks=[early_stopping, model_checkpoint2])
history3 = model3.fit(train_padded, train_labels2, epochs=num_epochs, batch_size=batch_size,
                    validation_data=(val_padded, val_labels2), callbacks=[early_stopping, model_checkpoint3])

# Plot accuracy graph for history1
plt.plot(history1.history['accuracy'], label='Model 1 Training Accuracy')
plt.plot(history1.history['val_accuracy'], label='Model 1 Validation Accuracy')

# Plot accuracy graph for history2
plt.plot(history2.history['accuracy'], label='Model 2 Training Accuracy')
plt.plot(history2.history['val_accuracy'], label='Model 2 Validation Accuracy')

# Plot accuracy graph for history3
plt.plot(history3.history['accuracy'], label='Model 3 Training Accuracy')
plt.plot(history3.history['val_accuracy'], label='Model 3 Validation Accuracy')

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Load the best model
model1 = tf.keras.models.load_model('rnn1')
model2 = tf.keras.models.load_model('rnn2')
model3 = tf.keras.models.load_model('rnn3')

# Evaluate on the test set
print("Model 1")
test_loss, test_accuracy = model1.evaluate(val_padded, val_labels1)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

print("Model 2")
test_loss, test_accuracy = model2.evaluate(val_padded, val_labels2)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

print("Model 3")
test_loss, test_accuracy = model3.evaluate(val_padded, val_labels3)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Make predictions on the test set
test_predictions1 = model1.predict(val_padded)
test_predictions2 = model2.predict(val_padded)
test_predictions3 = model3.predict(val_padded)
# Convert probabilities to binary predictions using a threshold (e.g., 0.5)
predicted_labels1 = (test_predictions1 > 0.5).astype(int)
predicted_labels2 = (test_predictions2 > 0.5).astype(int)
predicted_labels3 = (test_predictions3 > 0.5).astype(int)

# Apply the rules to generate predicted_labels4
predicted_labels4 = []

for i in range(len(predicted_labels1)):
    label1 = predicted_labels1[i]
    label2 = predicted_labels2[i]
    label3 = predicted_labels3[i]

    if np.logical_and(np.logical_and(label1 == 1, label2 == 0), label3 == 0):
        predicted_labels4.append(0)
    elif np.logical_and(np.logical_and(label1 == 0, label2 == 1), label3 == 0):
        predicted_labels4.append(1)
    elif np.logical_and(np.logical_and(label1 == 0, label2 == 0), label3 == 1):
        predicted_labels4.append(2)
    elif np.logical_and(np.logical_and(label1 == 1, label2 == 1), label3 == 0):
        predicted_labels4.append(3)
    elif np.logical_and(np.logical_and(label1 == 1, label2 == 0), label3 == 1):
        predicted_labels4.append(4)
    elif np.logical_and(np.logical_and(label1 == 0, label2 == 1), label3 == 1):
        predicted_labels4.append(5)
    elif np.logical_and(np.logical_and(label1 == 1, label2 == 1), label3 == 1):
        predicted_labels4.append(6)
    else:
        predicted_labels4.append(7)  

# Convert the list to numpy array
predicted_labels4 = np.array(predicted_labels4)

# Calculate accuracy for predicted_labels4
accuracy4 = accuracy_score(val_labels, predicted_labels4)

print(f"Accuracy for predicted_labels4: {accuracy4 * 100:.2f}%")


# Calculate additional metrics if needed
precision = precision_score(val_labels, predicted_labels4, average='weighted')
recall = recall_score(val_labels, predicted_labels4, average='weighted')
f1 = f1_score(val_labels, predicted_labels4, average='weighted')

print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1 Score: {f1 * 100:.2f}%")

# Print confusion matrix
conf_mat = confusion_matrix(val_labels, predicted_labels4)

# Display confusion matrix using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=sorted(set(train_labels)), yticklabels=sorted(set(train_labels)))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Create a DataFrame with the original text, cleaned text, predicted label, and true label
results_df = pd.DataFrame({
    'Original Text': val_ori_text,
    'Predicted Label': predicted_labels4,
    'True Label': val_labels,
})

# Save the DataFrame to a CSV file
results_df.to_csv('results\\test_results.csv', index=False)
