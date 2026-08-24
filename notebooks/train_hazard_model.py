import pandas as pd
import statsmodels.api as sm

print("--- LOADING MASTER DATASET ---")
file_path = "../outputs/MASTER_MODEL_DATASET.csv"
df = pd.read_csv(file_path)

# Drop any rows with NaN values resulting from the lag calculations
df = df.dropna().copy()

print(f"Dataset loaded. Training on {len(df)} monthly observations.")

# 1. Define the Target Variable (y)
y = df['Admission_Surge_Flag']

# 2. Define the Feature Matrix (X)
# We use the lagged variables to ensure we are forecasting future risk based on past data
features = ['CPI_3mo_lag', 'Rent_3mo_lag', 'Simcoe_Poverty_Rate']
X = df[features]

# Add a constant (intercept) for the statsmodels regression
X = sm.add_constant(X)

print("\n--- FITTING DISCRETE-TIME LOGISTIC HAZARD MODEL ---")
# 3. Fit the Logistic Regression model
try:
    hazard_model = sm.Logit(y, X).fit(disp=False)
    print("\n--- MODEL SUMMARY ---")
    print(hazard_model.summary())
except Exception as e:
    print(f"Model fitting failed. Error: {e}")

# 4. Generate Forecast Probabilities
# This generates the % risk of a surge for each row based on the economic inputs
df['Surge_Probability_Forecast'] = hazard_model.predict(X)

# 5. Save the Scored Dataset for Dashboarding
output_path = "../outputs/SCORED_SHELTER_FORECAST.csv"
df.to_csv(output_path, index=False)

print(f"\nSuccess! Scored forecast dataset saved to {output_path}")
print(df[['ShelterName_Clean', 'YearMonth', 'Admission_Surge_Flag', 'Surge_Probability_Forecast']].head())