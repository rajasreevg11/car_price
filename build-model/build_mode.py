import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import joblib

df = pd.read_csv("carprice_og.csv")

df.isnull().sum()
df.dropna(inplace=True)

df.drop("Model", axis=1, inplace=True)
df["Engine_cc"] = df["Engine"].str.extract("(\d+)").astype(int)
df[["Power_bhp", "Power_rpm"]] = df["Max Power"].str.extract(
    "(\d+) bhp @ (\d+) rpm"
)
df["Power_bhp"] = pd.to_numeric(df["Power_bhp"])
df["Power_rpm"] = pd.to_numeric(df["Power_rpm"])
df[["Torque_Nm", "Torque_rpm"]] = df["Max Torque"].str.extract(
    "(\d+) Nm @ (\d+) rpm"
)
df["Torque_Nm"] = pd.to_numeric(df["Torque_Nm"])
df["Torque_rpm"] = pd.to_numeric(df["Torque_rpm"])
df.drop(columns=["Max Power", "Max Torque", "Engine"], inplace=True, axis=1)
fuel = {
    "Petrol": 1,
    "Diesel": 2,
    "CNG": 3,
    "LPG": 4,
    "Electric": 5,
    "CNG + CNG": 6,
    "Hybrid": 7,
    "Petrol + CNG": 8,
    "Petrol + LPG": 9,
}
df["Fuel Type"] = df["Fuel Type"].map(fuel)
trans = {"Manual": 1, "Automatic": 2}
df["Transmission"] = df["Transmission"].map(trans)
owner = {
    "First": 1,
    "Second": 2,
    "Third": 3,
    "UnRegistered Car": 4,
    "4 or More": 5,
}
df["Owner"] = df["Owner"].map(owner)
seller = {"Individual": 1, "Corporate": 2, "Commercial Registration": 3}
df["Seller Type"] = df["Seller Type"].map(seller)
drivetrain = {"FWD": 1, "RWD": 2, "AWD": 3}
df["Drivetrain"] = df["Drivetrain"].map(drivetrain)

# Example: Encoding categorical columns
cols = ["Make", "Owner", "Fuel Type", "Location", "Color"]
label_encoders = {}

for col in cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Example: Handling missing values
df.fillna(df.mean(), inplace=True)

# Example: Train a linear regression model
X = df[["Make", "Owner", "Year", "Kilometer", "Fuel Type"]]
y = df["Price"]

model = LinearRegression()
model.fit(X, y)


# save the model using joblib
joblib.dump(model, 'lr_model.joblib')
