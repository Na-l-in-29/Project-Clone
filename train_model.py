import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

# Ensure the 'model' directory exists
os.makedirs('./model', exist_ok=True)

# Load the dataset
try:
    file_path = './data/predictive_maintenance.csv'
    data = pd.read_csv(file_path)
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: Dataset file not found. Please check the file path.")
    exit()

# Encode categorical columns
label_encoders = {}
categorical_columns = data.select_dtypes(include=['object']).columns
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le  # Save encoders for inverse transformations if needed
print(f"Categorical columns encoded: {list(categorical_columns)}")

# Save feature names for consistency
feature_columns = data.drop(['Target', 'UDI'], axis=1).columns  # Drop 'Target' and 'UDI'
joblib.dump(list(feature_columns), './model/feature_columns.pkl')
print(f"Feature columns saved: {list(feature_columns)}")

# Splitting the dataset into features (X) and target (y)
X = data[feature_columns]
y = data['Target']

# Splitting into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Dataset split into training and test sets.")

# Train a Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
print("Model training completed!")

# Evaluate the model
y_pred = model.predict(X_test)
print("Model evaluation report:")
print(classification_report(y_test, y_pred))

# Save the model
model_path = './model/trained_model.pkl'
joblib.dump(model, model_path)
print(f"Trained model saved at {model_path}")

# Save label encoders for decoding predictions
encoder_path = './model/label_encoders.pkl'
joblib.dump(label_encoders, encoder_path)
print(f"Label encoders saved at {encoder_path}")
