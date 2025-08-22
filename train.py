import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import pickle

# 1. Load dataset
df = pd.read_csv("water_potability (1).csv")

# 2. Handle missing values
df = df.dropna()

# 3. Features and target
X = df.drop("Potability", axis=1)
y = df["Potability"]

# 4. Balance the dataset (Upsampling minority class)
df_majority = df[df.Potability == 0]
df_minority = df[df.Potability == 1]

df_minority_upsampled = resample(
    df_minority,
    replace=True,     # sample with replacement
    n_samples=len(df_majority), # match majority
    random_state=42
)

df_balanced = pd.concat([df_majority, df_minority_upsampled])

X = df_balanced.drop("Potability", axis=1)
y = df_balanced["Potability"]

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. Train RandomForest
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# 8. Save model and scaler
with open("rf_model.pkl", "wb") as f:
    pickle.dump(rf_model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ Model and Scaler saved successfully as rf_model.pkl and scaler.pkl")
