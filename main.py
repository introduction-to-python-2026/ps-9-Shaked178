import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('parkinsons.csv')

X = data.drop(['status', 'name'], axis=1)
y = data['status']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier(random_state=42)
model.fit(X_scaled, y)

joblib.dump(model, 'model.joblib')
joblib.dump(scaler, 'scaler.joblib')

print("Done")
