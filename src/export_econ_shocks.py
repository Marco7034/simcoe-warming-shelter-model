import pandas as pd
import numpy as np
import os

print("--- EXPORTING PUBLIC ECONOMIC SHOCKS ---")

# 1. Create the timeline
dates = pd.date_range(start='2023-01-01', end='2026-04-01', freq='MS')
shock_matrix = pd.DataFrame({'Date': dates})
shock_matrix['YearMonth'] = shock_matrix['Date'].dt.strftime('%Y-%m')
shock_matrix['Year'] = shock_matrix['Date'].dt.year

# 2. Add Economic Data (Using the simulated data until you map the real pipeline outputs)
np.random.seed(42)
shock_matrix['Ontario_CPI'] = np.random.uniform(150, 160, len(shock_matrix))

rent_data = pd.DataFrame({
    'Year': [2023, 2024, 2025, 2026],
    'Barrie_Avg_Rent': [1800, 1950, 2050, 2050]
})
shock_matrix = pd.merge(shock_matrix, rent_data, on='Year', how='left')

poverty_data = pd.DataFrame({
    'Year': [2023, 2024, 2025, 2026],
    'Simcoe_Poverty_Rate': [8.5, 8.5, 8.5, 8.5] 
})
shock_matrix = pd.merge(shock_matrix, poverty_data, on='Year', how='left')

# 3. Create the Lags
shock_matrix['CPI_3mo_lag'] = shock_matrix['Ontario_CPI'].shift(3)
shock_matrix['Rent_3mo_lag'] = shock_matrix['Barrie_Avg_Rent'].shift(3)

# 4. Filter to the model timeline and save
shock_matrix = shock_matrix[shock_matrix['YearMonth'] >= '2023-11'].copy()
output_path = "../outputs/PUBLIC_ECON_SHOCKS.csv"
shock_matrix.to_csv(output_path, index=False)

print(f"Success! Standalone economic matrix saved to {output_path}")