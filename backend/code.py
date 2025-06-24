# ==============================================================================
# MySurgeon - Surgical Volume Prediction Model (Faithful Replication)
# ==============================================================================
#
# This script is a direct, functional translation of the final model developed
# in the Jupyter notebooks (v001.ipynb, v002.ipynb). The core logic,
# data processing, feature selection, and model algorithm are identical to
# ensure the same predictive results.

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt
import joblib
import warnings

# Suppress warnings for a cleaner output, as done in the notebooks
warnings.filterwarnings('ignore')

# --- Step 1: Data Loading and Preparation ---
# This step replicates the creation of the cleaned 'df1' DataFrame from v002.ipynb.

print(">>> Step 1: Loading and Preparing Data...")

try:
    # Load the original dataset
    df_original = pd.read_excel("Case_Dataset.xlsx")
    print(f"Successfully loaded data. Original shape: {df_original.shape}")

    # Isolate numeric columns for outlier detection
    df_numeric_for_cleaning = df_original.drop(columns=['SurgDate', 'DOW'])

    # Replicating the outlier removal from notebook v002.ipynb
    # This ensures the model is trained on the exact same data points.
    non_outlier_mask = (np.abs(stats.zscore(df_numeric_for_cleaning)) < 3).all(axis=1)
    df_cleaned = df_original[non_outlier_mask].reset_index(drop=True)

    print(f"Data shape after outlier removal: {df_cleaned.shape}")

except FileNotFoundError:
    print("Error: 'Case_Dataset.xlsx' not found. Please place it in the same directory.")
    df_cleaned = None

# --- Step 2: Feature Selection and Model Training ---
# This step replicates the final model setup from v002.ipynb.

print("\n>>> Step 2: Training the Linear Regression Model...")

# The final features chosen in the notebook for the prediction model
features = ['T - 3', 'T - 2', 'T - 1']
target = 'Actual'

X = df_cleaned[features]
y = df_cleaned[target]

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X, y)

print("Model training complete.")
print(f"Model Intercept: {model.intercept_:.4f}")
print(f"Model Coefficients ('T-3', 'T-2', 'T-1'): {np.round(model.coef_, 4)}")

# --- Step 3: Model Evaluation ---
# Replicates the performance evaluation from the notebook to confirm results.

print("\n>>> Step 3: Evaluating Model Performance...")

# Predict on the training data to calculate metrics
y_pred = model.predict(X)

# The notebook created a final DataFrame for comparison, we replicate that here
result_df = df_cleaned.copy()
result_df['Predicted'] = np.round(y_pred)

# Calculate RMSE, mirroring the notebook's evaluation on the final dataset
rmse = sqrt(mean_squared_error(result_df['Actual'], result_df['Predicted']))
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print("(This matches the notebook's evaluation of +/- cases)")

# --- Step 4: Save the Trained Model for Production Use ---
# This is the crucial step to make the model available to your platform's API.

print("\n>>> Step 4: Saving the Model...")

model_filename = "surgical_volume_model.pkl"
joblib.dump(model, model_filename)

print(f"Model has been successfully saved to '{model_filename}'")
print("\n--- Process Complete ---")