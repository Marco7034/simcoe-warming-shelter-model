import pandas as pd
import statsmodels.api as sm
import os

print("--- LOADING MASTER DATASET ---")
file_path = "../outputs/MASTER_MODEL_DATASET.csv"

if not os.path.exists(file_path):
    print(f"CRITICAL ERROR: Could not find {file_path}")
    exit()

df = pd.read_csv(file_path)

# Ensure no completely blank rows crash the model
df.dropna(how='all', inplace=True)

# Define the exact features, including the new meteorological data
# Define the exact features, removing the constant Poverty Rate
features = [
    'CPI_3mo_lag', 
    'Rent_3mo_lag', 
    'Extreme_Min_Temp', 
    'Total_Snow_cm'
]
target = 'Admission_Surge_Flag'

# Drop any remaining NaNs specifically in our training columns to ensure mathematical stability
df_clean = df.dropna(subset=features + [target]).copy()

print(f"Dataset loaded. Training on {len(df_clean)} monthly observations.\n")

if len(df_clean) == 0:
    print("CRITICAL ERROR: 0 observations remaining after dropping NaNs. Check your data merges.")
    exit()

print("--- FITTING DISCRETE-TIME LOGISTIC HAZARD MODEL ---")
X = df_clean[features]
y = df_clean[target]

# Add a constant for the intercept
X = sm.add_constant(X)

try:
    # Fit the logistic regression model
    hazard_model = sm.Logit(y, X).fit(disp=False)
    
    print("\n--- MODEL SUMMARY ---")
    print(hazard_model.summary())
    
    # Generate the probability forecast
    df_clean['Surge_Probability_Forecast'] = hazard_model.predict(X)
    
    # Isolate the final output for PowerBI
    output_df = df_clean[['ShelterName_Clean', 'YearMonth', 'Admission_Surge_Flag', 'Surge_Probability_Forecast']]
    
    forecast_path = "../outputs/SCORED_SHELTER_FORECAST.csv"
    output_df.to_csv(forecast_path, index=False)
    
    print(f"\nSuccess! Scored forecast dataset saved to {forecast_path}")

except Exception as e:
    print(f"Model fitting failed. Error: {e}")
