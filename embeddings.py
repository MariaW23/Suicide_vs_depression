from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tqdm import tqdm
import torch

# Create train test split
data = pd.read_csv('data/data.csv')
# random state set to random int to make sure randomness is reproducible
# approximately 70-15-15 split
# train_data, test_data = train_test_split(data, test_size=0.15, random_state=42)
# train_data, validate_data = train_test_split(data, test_size=0.18, random_state=42)

# train_data.to_csv('data/train_data.csv', index=False)
# validate_data.to_csv('data/validate_data.csv', index=False)
# test_data.to_csv('data/test_data.csv', index=False)


# Download the model to local so it can be used repeatedly
# directory_path = 'content/module_useT'
# os.makedirs(directory_path, exist_ok=True)
# # Download the module, and uncompress it to the destination folder. 
# url = 'https://tfhub.dev/google/universal-sentence-encoder-large/3?tf-hub-format=compressed'
# command = f'curl -L "{url}" | tar -zxvC {directory_path}'

# Execute the command using subprocess
# subprocess.run(command, shell=True)


# train_data = pd.read_csv("data/train_data.csv")

# Google Universal Sentence Encoder (GUSE) Transformer

# get the GUSE model
GUSE_model = hub.load("https://www.kaggle.com/models/google/universal-sentence-encoder/TensorFlow2/large/2")

# reduce the logging output
tf.get_logger().setLevel('ERROR')

# split into batches so as computer can handle it
batch_size = 100
GUSE_embeddings_df = pd.DataFrame()

for start in range(0, len(data), batch_size):
    print(f"Batch {start//batch_size}...")
    end = min(start + batch_size, len(data))
    batch_texts = data["selftext"][start:end].astype(str).tolist()
    batch_texts_tensor = tf.convert_to_tensor(batch_texts, dtype=tf.string)
    batch_embeddings = GUSE_model(batch_texts_tensor)
    if tf.is_tensor(batch_embeddings):
        batch_embeddings = batch_embeddings.numpy()
    batch_embeddings_df = pd.DataFrame(batch_embeddings)
    GUSE_embeddings_df = pd.concat([GUSE_embeddings_df, batch_embeddings_df], ignore_index=True)

GUSE_embeddings_df.to_csv("data/guse_embeddings.csv", index=False)

# BERT Transformer
# from transformers import BertModel, BertTokenizer

# # initializes BERT's base-uncased style configuration (trained on lowercases for consistency, model tokenizer will convert text to lowercase before processing as well)
# model = BertModel.from_pretrained("google-bert/bert-base-uncased")
# tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")

# batch_size = 100
# embeddings_list = []
# for i in tqdm(range(0, len(data), batch_size)):
#     batch = data[i:i + batch_size].astype(str)
    
#     # Tokenize and pad the batch
#     tokenized = [tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=512) for text in batch]
#     tokenized_padded = tokenizer.pad({'input_ids': tokenized}, padding=True, return_tensors="pt")
    
#     input_ids = tokenized_padded['input_ids']
#     attention_mask = tokenized_padded['attention_mask']

#     # Use no_grad because no gradient is necessary
#     with torch.no_grad():
#         last_hidden_states = model(input_ids, attention_mask=attention_mask)
    
#     # Get 2D BERT embeddings
#     embeddings = last_hidden_states.last_hidden_state[:,0,:]

#     # Convert embeddings to numpy array and append to the list
#     embeddings_list.append(embeddings.cpu().numpy())

# # Concatenate all the embeddings
# all_embeddings = np.concatenate(embeddings_list, axis=0)

# # Convert the embeddings to a DataFrame
# embeddings_df = pd.DataFrame(all_embeddings)

# # Save the DataFrame to a CSV file
# embeddings_df.to_csv('data/bert_embeddings.csv', index=False)

# def getFeatures(batch_1):
#     # sepcial tokens can include CLS (beginning of sequence), SEP (end of sequence), PAD (to pad to ensure equal length in sequence)
#     tokenized = batch_1.apply(lambda x: tokenizer.encode(x, add_special_tokens=True, truncation=True, max_length=512)).tolist()

#     tokenized_padded = tokenizer.pad({'input_ids': tokenized}, padding=True, return_tensors="pt")
    
#     input_ids = tokenized_padded['input_ids']
#     attention_mask = tokenized_padded['attention_mask']

#     # Use no_grad because no gradient is necessary
#     with torch.no_grad():
#         last_hidden_states = model(input_ids, attention_mask=attention_mask)
    
#     # get 2D bert embeddings
#     embeddings = last_hidden_states[0][:,0,:]

#     # get 3D bert embeddings
#     # embeddings = last_hidden_states[0].numpy() # use this line if you want the 3D BERT features 
#     return embeddings


# Sentense BERT
# from sentence_transformers import SentenceTransformer

# model = SentenceTransformer('all-MiniLM-L6-v2')

# batch_size = 100
# # To catch and fill the parts that are empty as string
# data = data["selftext"].fillna("").astype(str)
# embeddings_list = []
# for start in tqdm(range(0, len(data), batch_size)):
#     end = min(start + batch_size, len(data))
#     batch_texts = data[start:end].tolist()
#     batch_embeddings = model.encode(batch_texts, show_progress_bar=False)
#     embeddings_list.append(batch_embeddings)

# bert_embeddings = np.vstack(embeddings_list)
# np.savetxt("data/sbert_embeddings.csv", bert_embeddings, delimiter=",")
