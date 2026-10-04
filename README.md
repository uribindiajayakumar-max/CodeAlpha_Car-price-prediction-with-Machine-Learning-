import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
9/28/26, 7:44 PM CodeAlpha Data Science Internship Guide
https://gemini.google.com/app/a74e08b8eaed3b7b 3/7 df = pd.read_csv("car data.csv")
# Feature Engineering: Calculate car age
current_year = 2026
df["Car_Age"] = current_year - df["Year"]
df = df.drop(columns=["Car_Name", "Year"])
# One-hot encode categorical variables (Fuel_Type, Seller_Type, Transmission)
df = pd.get_dummies(df, drop_first=True)
# Train Test Split
X = df.drop(columns=["Selling_Price"])
y = df["Selling_Price"]
X_train, X_test, y_train, y_test = train_test_split(
 X, y, test_size=0.2, random_state=42
)
# Train Model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)
# Predictions & Visualization
y_pred = model.predict(X_test)
print("R2 Score:", r2_score(y_test, y_pred))
plt.scatter(y_test, y_pred, color="blue", alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--") # Ideal line
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Selling Prices")
plt.show()
Task 4: Sales Prediction using Python (Regression)