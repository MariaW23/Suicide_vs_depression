import numpy as np
import pandas as pd

original_train_labels = pd.read_csv("data/train_data.csv")["is_suicide"]
original_test_labels = pd.read_csv("data/test_data.csv")["is_suicide"]

original_train_labels = np.asarray(original_train_labels)
original_test_labels = np.asarray(original_test_labels)

predicted_labels = pd.read_csv("data/clustering_results.csv")["predictions"]
predicted_probs = pd.read_csv("data/clustering_results.csv")["confidence"]

predicted_labels = np.asarray(predicted_labels)
predicted_probs = np.asarray(predicted_probs)

predicted_train_labels = predicted_labels[0:len(original_train_labels)]
predicted_train_probs = predicted_probs[:len(original_train_labels)]

predicted_test_labels = predicted_labels[len(original_train_labels):]
predicted_test_probs = predicted_probs[len(original_test_labels):]

# Threshold for label correction
tau = 0.90

# Correcting training labels
final_train_labels = []
for i in range(len(original_train_labels)):
    if original_train_labels[i] != predicted_train_labels[i]:
        if predicted_train_probs[i] > tau or predicted_train_probs[i] < (1-tau):
            final_train_labels.append(predicted_train_labels[i])
        else:
            final_train_labels.append(original_train_labels[i])
    else:
        final_train_labels.append(original_train_labels[i])
    
# final_train_labels = np.asarray(final_train_labels)


# Adding corrected train labels to train dataset
train_data = pd.read_csv("data/train_data.csv")
# Ensure the length of the list matches the number of rows in the DataFrame
if len(final_train_labels) != len(train_data):
    raise ValueError("Length of corrected labels invalid")

train_data["corrected"] = final_train_labels
train_data.to_csv("data/corrected_train_data.csv", index=False)
