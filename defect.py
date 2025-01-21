import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

# Load the dataset
data = pd.read_csv('data/predictive_maintenance.csv')

# Drop irrelevant columns
data = data.drop(columns=["UDI", "Product ID"])

# Define features and target
X = data.drop(columns=["Target", "Failure Type"])
y = data["Target"]

# Identify categorical and numerical columns
categorical_cols = ["Type"]
numerical_cols = [col for col in X.columns if col not in categorical_cols]

# Preprocessing: Scaling and One-Hot Encoding
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(drop="first", handle_unknown='ignore'), categorical_cols),
    ]
)

# Preprocess the features
preprocessor.fit(X)  
X_preprocessed = preprocessor.transform(X)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_preprocessed, y)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# Define and train the model
model = LogisticRegression(random_state=42, max_iter=1000, class_weight="balanced")
model.fit(X_train, y_train)

# Save the trained model and preprocessor
with open("model/downtime_model.pkl", "wb") as file:
    pickle.dump({"model": model, "preprocessor": preprocessor}, file)

print("\nTrained model saved as 'downtime_model.pkl'")
