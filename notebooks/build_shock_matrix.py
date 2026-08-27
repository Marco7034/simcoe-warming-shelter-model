import pandas as pd
import numpy as np
import glob
import os

print("--- BUILDING MASTER DATASET (WEATHER & ECON) ---")

# 1. LOAD SHELTER DATA
print("Loading shelter panel...")
try:
    shelter_df = pd.read_csv("../data_internal/FINAL_MODEL_TRAIN.csv")
    # Ensure YearMonth is a string
    shelter_df['YearMonth'] = shelter_df['YearMonth'].astype(str)
except Exception as e:
    print(f"CRITICAL ERROR loading shelter data: {e}")
    exit()

# 2. AGGREGATE WEATHER DATA
print("Processing high-frequency meteorological data...")
weather_files = glob.glob("../data_public/raw_weather/*.csv")
weather_list = []

for f in weather_files:
    try:
        df = pd.read_csv(f, low_memory=False)
        weather_list.append(df)
    except Exception as e:
        print(f"Skipping {f}: {e}")

if weather_list:
    weather_df = pd.concat(weather_list, ignore_index=True)
    # Create matching YearMonth column
    weather_df['YearMonth'] = weather_df['Year'].astype(str) + '-' + weather_df['Month'].astype(str).str.zfill(2)
    
    # Compress daily readings into monthly hazard triggers
    monthly_weather = weather_df.groupby('YearMonth').agg({
        'Min Temp (°C)': 'min',          # Coldest day of the month
        'Mean Temp (°C)': 'mean',        # Average temperature
        'Total Snow (cm)': 'sum'         # Cumulative snowfall
    }).reset_index()
    
    monthly_weather.rename(columns={
        'Min Temp (°C)': 'Extreme_Min_Temp',
        'Mean Temp (°C)': 'Average_Mean_Temp',
        'Total Snow (cm)': 'Total_Snow_cm'
    }, inplace=True)
else:
    print("WARNING: No weather files found in data_public/raw_weather/")
    monthly_weather = pd.DataFrame(columns=['YearMonth', 'Extreme_Min_Temp', 'Average_Mean_Temp', 'Total_Snow_cm'])

# 3. PARSE MACROECONOMIC DATA
print("Parsing external economic metrics...")
# Ontario CPI
try:
    with open("../data_public/raw_econ/ontario_cpi_monthly.csv", 'r', encoding='utf-8') as f:
        lines = f.readlines()
    header_idx = [i for i, line in enumerate(lines) if "November 2023" in line][0]
    
    cpi_raw = pd.read_csv("../data_public/raw_econ/ontario_cpi_monthly.csv", skiprows=header_idx)
    cpi_all = cpi_raw[cpi_raw.iloc[:, 0].astype(str).str.contains('All-items', na=False)]
    
    cpi_melted = cpi_all.melt(id_vars=cpi_raw.columns[0], var_name='MonthYear', value_name='Ontario_CPI')
    cpi_melted['YearMonth'] = pd.to_datetime(cpi_melted['MonthYear'], format='%B %Y', errors='coerce').dt.strftime('%Y-%m')
    cpi_clean = cpi_melted.dropna(subset=['YearMonth'])[['YearMonth', 'Ontario_CPI']].copy()
    cpi_clean['Ontario_CPI'] = pd.to_numeric(cpi_clean['Ontario_CPI'], errors='coerce')
except Exception as e:
    print("WARNING: Could not parse Ontario CPI perfectly. Using fallback.")
    cpi_clean = pd.DataFrame(columns=['YearMonth', 'Ontario_CPI'])

# Barrie Rent
try:
    csd = pd.read_csv("../data_public/raw_econ/CSD_202410_CDP.csv", low_memory=False, encoding='latin1')
    barrie_rent = csd[
        (csd['CSDNAME'] == 'Barrie') & 
        (csd['bedroom_type'] == '2 Bedrooms') & 
        (csd['dwelling_type'] == 'Apartment') &
        (csd['construction_year_range'] == 'Overall') &
        (csd['size_in_units_range'] == 'Overall') &
        (csd['rent_range'] == 'Overall')
    ]
    rent_clean = barrie_rent[['Year', 'average_rent']].rename(columns={'average_rent': 'Barrie_Avg_Rent'}).dropna()
    rent_clean['Year'] = rent_clean['Year'].astype(str)
except Exception as e:
    print("WARNING: Could not parse Barrie Rent perfectly. Using fallback.")
    rent_clean = pd.DataFrame(columns=['Year', 'Barrie_Avg_Rent'])

# 4. THE GRAND MERGE
print("Merging feature space...")
# Merge Weather
master_df = pd.merge(shelter_df, monthly_weather, on='YearMonth', how='left')

# Merge CPI
master_df = pd.merge(master_df, cpi_clean, on='YearMonth', how='left')

# Extract Year from YearMonth to merge annual rent/poverty data
master_df['MergeYear'] = master_df['YearMonth'].str.slice(0, 4)

# Merge Rent
master_df = pd.merge(master_df, rent_clean, left_on='MergeYear', right_on='Year', how='left')

# Drop the temporary merge column
master_df.drop(columns=['MergeYear', 'Year_y'], inplace=True, errors='ignore')
if 'Year_x' in master_df.columns:
    master_df.rename(columns={'Year_x': 'Year'}, inplace=True)

# 5. ENGINEER LAGS
print("Engineering 3-month predictive lags...")
master_df.sort_values(by=['ShelterName_Clean', 'YearMonth'], inplace=True)

master_df['CPI_3mo_lag'] = master_df.groupby('ShelterName_Clean')['Ontario_CPI'].shift(3)
master_df['Rent_3mo_lag'] = master_df.groupby('ShelterName_Clean')['Barrie_Avg_Rent'].shift(3)
master_df['Snow_1mo_lag'] = master_df.groupby('ShelterName_Clean')['Total_Snow_cm'].shift(1)

# Backfill any missing lags strictly for the model matrix
master_df.bfill(inplace=True)

output_path = "../outputs/MASTER_MODEL_DATASET.csv"
master_df.to_csv(output_path, index=False)

print(f"Success! Fully mapped Master Model Dataset saved to {output_path}")
print(master_df[['ShelterName_Clean', 'YearMonth', 'Extreme_Min_Temp', 'Total_Snow_cm', 'CPI_3mo_lag']].head())
