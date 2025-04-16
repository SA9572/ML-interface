# train_model.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Step 1: Create or Load Data
# Synthetic data (replace this with your real dataset)
data = {
    'tensile_strength': [200, 300, 150, 350, 250, 180, 320, 270, 140, 290],
    'proof_stress': [150, 250, 100, 280, 200, 130, 260, 220, 90, 240],
    'elongation': [5, 15, 3, 20, 10, 4, 18, 12, 2, 14],
    'quality': [0, 1, 0, 1, 1, 0, 1, 1, 0, 1]  # 1 = Good, 0 = Not Suitable
}

df = pd.DataFrame(data)

# Features (X) and target (y)
X = df[['tensile_strength', 'proof_stress', 'elongation']]
y = df['quality']

# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 4: Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Step 5: Save the trained model
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model saved as 'model.pkl'")

# Optional: Test the model with a sample input
sample_input = np.array([[300, 250, 15]])  # Example: tensile=300, proof=250, elongation=15
prediction = model.predict(sample_input)
print(f"Sample Prediction: {'Good' if prediction[0] == 1 else 'Not Suitable'}")