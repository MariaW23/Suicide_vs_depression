from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


# Create train test split
# data = pd.read_csv('data/suicide_vs_depression.csv')
# # random state set to random int to make sure randomness is reproducible
# train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
# train_data.to_csv('data/train_data.csv', index=False)
# test_data.to_csv('data/test_data.csv', index=False)


# Download the model to local so it can be used repeatedly
# directory_path = 'content/module_useT'
# os.makedirs(directory_path, exist_ok=True)
# # Download the module, and uncompress it to the destination folder. 
# url = 'https://tfhub.dev/google/universal-sentence-encoder-large/3?tf-hub-format=compressed'
# command = f'curl -L "{url}" | tar -zxvC {directory_path}'

# Execute the command using subprocess
# subprocess.run(command, shell=True)


train_data = pd.read_csv("data/train_data.csv")

# Google Universal Sentence Encoder (GUSE) Transformer

# get the GUSE model
GUSE_model = hub.load("https://www.kaggle.com/models/google/universal-sentence-encoder/TensorFlow2/large/2")

# reduce the logging output
tf.get_logger().setLevel('ERROR')

# split into batches so as computer can handle it
batch_size = 100
GUSE_embeddings_df = pd.DataFrame()

for start in range(0, len(train_data), batch_size):
    print(f"Batch {start//batch_size}...")
    end = min(start + batch_size, len(train_data))
    batch_texts = train_data["selftext"][start:end]
    batch_embeddings = GUSE_model(batch_texts)
    if tf.is_tensor(batch_embeddings):
        batch_embeddings = batch_embeddings.numpy()
    batch_embeddings_df = pd.DataFrame(batch_embeddings)
    GUSE_embeddings_df = pd.concat([GUSE_embeddings_df, batch_embeddings_df], ignore_index=True)

GUSE_embeddings_df.to_csv("data/guse-embeddings.csv")
