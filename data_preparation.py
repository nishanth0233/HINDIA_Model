import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
dataset_path = 'hindi_dataset.csv'  # Update the file path

# Assuming the file is a plain text file with one sentence per line
data = pd.read_csv(dataset_path, header=None, names=['sentence'], sep="\n")

# Basic Preprocessing: Assuming the dataset is already clean and only contains Hindi characters and standard punctuation

# Split the data into training and testing sets
train_data, test_data = train_test_split(data['sentence'], test_size=0.2, random_state=42)

# Save the training and testing sets to CSV files
train_data.to_csv('train_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)

# Display the first few rows of the split to verify
print("Training Data Sample:")
print(train_data.head())
print("\nTesting Data Sample:")
print(test_data.head())
