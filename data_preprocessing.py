import pandas as pd

# Load the dataset
data = pd.read_csv('data/predictive_maintenance.csv')

# Display dataset information
print("Dataset Preview:")
print(data.head())
print("\nDataset Information:")
print(data.info())

# Save the processed data to use in the model script if needed
data.to_csv('data/processed_data.csv', index=False)