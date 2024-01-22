import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load your training, validation, and testing data
all_data = pd.read_csv("esgn_all.csv", encoding= 'unicode_escape')
train_data = pd.read_csv("train2.csv")
val_data = pd.read_csv("val2.csv")
test_data = pd.read_csv("test2.csv")


# Explore class distribution
plt.figure(figsize=(12, 6))
sns.countplot(x='class', data=all_data , palette='viridis')
plt.title('Class Distribution in All data ')
plt.show()

# # Explore text length distribution
# all_data['text_length'] = all_data ['ESGN'].apply(lambda x: len(x.split()))
# plt.figure(figsize=(12, 6))
# sns.histplot(data=all_data , x='text_length', bins=30, kde=True, color='blue')
# plt.title('Distribution of Text Length in Training Data')
# plt.xlabel('Text Length')
# plt.show()


# Combine all text data for word cloud
all_text = ' '.join(all_data ['ESGN'])

# Generate a word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)

# Display the word cloud using matplotlib
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for Training Data')
plt.show()
