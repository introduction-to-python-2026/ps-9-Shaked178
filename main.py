import pandas as pd
import joblib
import yaml

# טעינת קובץ הקונפיג
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# טעינת המודל
model = joblib.load(config["path"])

# טעינת דאטה חדש לחיזוי
data = pd.read_csv("parkinsons.csv")
X = data[config["features"]]

# חיזוי
predictions = model.predict(X)

# הצגת התוצאות
print(predictions)
