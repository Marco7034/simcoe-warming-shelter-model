import pandas as pd
import numpy as np
import os

print("--- BUILDING EXTERNAL SHOCK MATRIX ---")

# 1. Create a base timeline (Starting Jan 2023 to allow calculating 3-to-6 month lags)
dates = pd.date_range(start='2023-01-01', end='2026-04-01', freq='MS')
shock_matrix = pd.DataFrame({'Date': dates})
shock_matrix['YearMonth'] = shock_matrix['Date'].dt.strftime('%Y-%m')
shock_matrix['Year'] = shock_matrix['Date'].dt.year

# 2. Simulate CPI (Until you map the exact columns from ontario_cpi_monthly.csv)
np.random.seed(42)
shock_matrix['Ontario_CPI'] = np.random.uniform(150, 160, len(shock_matrix))

# 3. Process CMHC Rent (Annual -> Forward Filled to Monthly)
rent_data = pd.DataFrame({
    'Year': [2023, 2024, 2025, 2026],
    'Barrie_Avg_Rent': [1800, 1950, 2050, 2050]
})
shock_matrix = pd.merge(shock_matrix, rent_data, on='Year', how='left')

# 4. Process Working Poverty (Annual -> Forward Filled to Monthly)
# We will use a static rate here for the dummy test, but you can hook this up 
# directly to your simcoe_working_poverty_clean.xlsx file later.
poverty_data = pd.DataFrame({
    'Year': [2023, 2024, 2025, 2026],
    'Simcoe_Poverty_Rate': [8.5, 8.5, 8.5, 8.5] 
})
shock_matrix = pd.merge(shock_matrix, poverty_data, on='Year', how='left')

print("--- ENGINEERING LAG FEATURES ---")
# 5. Create Time-Lagged Features for the Hazard Model
shock_matrix['CPI_3mo_lag'] = shock_matrix['Ontario_CPI'].shift(3)
shock_matrix['CPI_6mo_lag'] = shock_matrix['Ontario_CPI'].shift(6)
shock_matrix['Rent_3mo_lag'] = shock_matrix['Barrie_Avg_Rent'].shift(3)

# Calculate economic momentum (e.g., how much did inflation jump in the last 3 months?)
shock_matrix['CPI_Momentum_3mo'] = shock_matrix['Ontario_CPI'] - shock_matrix['CPI_3mo_lag']

# Drop the buffer months we only used to calculate the lags (Keep Nov 2023 onward)
shock_matrix = shock_matrix[shock_matrix['YearMonth'] >= '2023-11'].copy()

print("--- MERGING WITH SHELTER PANEL ---")
# 6. Load the final shelter panel we just generated
shelter_panel_path = "../outputs/MODEL_TRAINING_PANEL_FINAL.csv"
try:
    shelter_panel = pd.read_csv(shelter_panel_path)
except FileNotFoundError:
    print(f"CRITICAL ERROR: Could not find '{shelter_panel_path}'.")
    exit()

# 7. The Grand Merge
master_dataset = pd.merge(shelter_panel, shock_matrix, on='YearMonth', how='left')

# 8. Save the final model-ready dataset
output_path = "../outputs/MASTER_MODEL_DATASET.csv"
master_dataset.to_csv(output_path, index=False)

print(f"Success! Master Model Dataset saved to {output_path}")
print(master_dataset[['ShelterName_Clean', 'YearMonth', 'Admission_Surge_Flag', 'CPI_3mo_lag', 'Rent_3mo_lag']].head())