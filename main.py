import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

df = pd.read_csv(cfg["data"])

X = df[cfg["features"]]
y = df[cfg["target"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

joblib.dump(model, cfg["model_path"])

preds = model.predict(X)
df["predicted_status"] = preds

print(df.head())

