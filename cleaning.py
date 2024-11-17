# Import the necessary modules
from transformers import RobertaTokenizer
import torch
import re
import emoji
import unicodedata
from nltk.corpus import stopwords

# Load the RoBERTa tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Text preprocessing function
# def preprocess_text(sen):
#     sentence = unicodedata.normalize('NFKD', sen)
#     sentence = sentence.lower()
#     sentence = re.sub(r'[^a-zA-Z]', ' ', sentence)
#     sentence = re.sub(r'\s+[a-zA-Z]\s+', ' ', sentence)
#     sentence = re.sub(r'\s+', ' ', sentence)
#     stop_words = set(stopwords.words('english'))
#     words = sentence.split()
#     filtered_words = [word for word in words if word not in stop_words]
#     sentence = ' '.join(filtered_words)
#     sentence = emoji.replace_emoji(sentence, replace='')
#     return sentence

def preprocess_text(sen):
    sentence = unicodedata.normalize('NFKD', sen)
    sentence = sentence.lower()
    # Keep punctuation and basic structure
    sentence = re.sub(r'[^a-zA-Z\s]', '', sentence)
    sentence = re.sub(r'\s+', ' ', sentence)
    return sentence


# Function to process and tokenize the review using RoBERTa tokenizer
def process_review(review):
    # Preprocess the text
    review = preprocess_text(review)
    
    # Tokenize the review using RoBERTa tokenizer
    inputs = tokenizer(review, max_length=256, padding='max_length', truncation=True, return_tensors='pt')
    
    # Return tokenized input
    return inputs
