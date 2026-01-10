import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
import yaml

# 1. Load the dataset
df = pd.read_csv("parkinsons.csv")

# 2. Select features and target
X = df[["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)"]]
y = df["status"]
selected_features = ["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)"]

print("X sample:")
print(X.head())
print("\nY sample:")
print(y.head())

# 3. Scale the data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=selected_features)

print("After scaling:")
print(X_scaled.head())

# 4. Split the data
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print("Train shape:", X_train.shape)
print("Validation shape:", X_val.shape)

# 5. Choose a model
model = SVC(kernel='rbf', C=167, gamma=300)  # הערכים שלך, אם רציתם לשפר דיוק
print("Selected model:", model)

# 6. Train and test the model
model.fit(X_train, y_train)
y_pred = model.predict(X_val)

accuracy = accuracy_score(y_val, y_pred)
print("Validation Accuracy:", accuracy)

if accuracy >= 0.8:
    print("✅ Accuracy requirement met!")
else:
    print("⚠ Accuracy below 0.8, consider tuning the model.")

# 7. Save the model and update config.yaml
model_filename = "my_model.joblib"
joblib.dump(model, model_filename)

config = {
    "selected_features": selected_features,
    "path": model_filename
}

with open("config.yaml", "w") as file:
    yaml.dump(config, file)

print("Model saved and config.yaml updated!")

