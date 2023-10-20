import pandas as pd
import numpy as np

# Load the dataset
data = pd.read_csv("/content/insurance.csv")

# Data Preprocessing and Exploration
# Check for missing values
missing_values = data.isnull().sum()

# Encode categorical variables using one-hot encoding
data = pd.get_dummies(data, columns=["sex", "smoker", "region"], drop_first=True)

# Explore the dataset
print(data.head())
print(data.describe())

import matplotlib.pyplot as plt
import seaborn as sns

# Visualizations
# Correlation matrix
correlation_matrix = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()


# Pairplot for selected features
selected_features = ["age", "bmi", "children", "smoker_yes", "charges"]
sns.pairplot(data[selected_features])
plt.show()

# Split the data into features (X) and the target variable (y)
X = data.drop(columns=["charges"])
y = data["charges"]

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Split the data into training and testing sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Build and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R-squared:", r2)

# Visualize the actual vs. predicted charges
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("Actual vs. Predicted Medical Insurance Charges")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', linewidth=2)
plt.show()

# Test the trained model with a new data point
new_data_point = {
    "age": 30,
    "sex_male": 0,  # Encode as 0 for female
    "bmi": 28.5,
    "children": 2,
    "smoker_yes": 1,  # Encode as 1 for "yes"
    "region_northwest": 0,  # Encode as 0 for other regions
    "region_southeast": 0,
    "region_southwest": 1,
}

# Create a DataFrame with the new data point
new_data_df = pd.DataFrame([new_data_point])

# Reorder the columns to match the order used during training
new_data_df = new_data_df[X.columns]

# Use the trained model to make a prediction
predicted_charges = model.predict(new_data_df)

# Print the predicted charges
print("Predicted Charges:", predicted_charges[0])
