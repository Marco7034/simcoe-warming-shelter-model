import os
import numpy as np
import pandas as pd

print("--- PROCESSING CMHC RENTAL MARKET DATA (2024 & 2025) ---")
# 1. Load 2024 CMHC files
met_2024 = pd.read_csv(
    "../data_public/raw_econ/Met_202410_CDP.csv", encoding="utf-8-sig"
)

csd_2024 = pd.read_csv(
    "../data_public/raw_econ/CSD_202410_CDP.csv", encoding="latin1"
)

barrie_met_24 = met_2024[
    met_2024["METNAME"].str.contains("Barrie", case=False, na=False)
]
simcoe_csd_24 = csd_2024[
    csd_2024["CSDNAME"].str.contains(
        "Orillia|Midland|Barrie|Simcoe", case=False, na=False
    )
]

# 2. Load 2025 CMHC files
met_2025 = pd.read_excel(
    "../data_public/raw_econ/Met_202510_CDP.xlsx", sheet_name="Met"
)
csd_2025 = pd.read_excel(
    "../data_public/raw_econ/CSD_202510_CDP.xlsx", sheet_name="CSD"
)

barrie_met_25 = met_2025[
    met_2025["METNAME"].str.contains("Barrie", case=False, na=False)
]
simcoe_csd_25 = csd_2025[
    csd_2025["CSDNAME"].str.contains(
        "Orillia|Midland|Barrie|Simcoe", case=False, na=False
    )
]

print(
    "Loaded CMHC rental rows -> 2024 Barrie Met:"
    f" {len(barrie_met_24)}, 2025 Barrie Met: {len(barrie_met_25)}"
)


print("\n--- PROCESSING 2023 WORKING POVERTY (T1FF) DATA ---")
# 3. Load raw Working Poverty table
wp_raw = pd.read_excel(
    "../data_public/raw_econ/R25055 - Working Poor, 2023.xlsx",
    sheet_name="2023 - WP_Table_1",
    header=None,
)

# Search for text names OR Statistics Canada geographic codes (3543, 35430)
mask = (
    wp_raw.astype(str)
    .apply(lambda col: col.str.contains("Barrie|Simcoe|Orillia|Midland|3543|35430", case=False, na=False))
    .any(axis=1)
)
simcoe_wp = wp_raw[mask]

print(
    f"Extracted {len(simcoe_wp)} matching rows from the 2023 Working Poverty table."
)


print("\n--- PROCESSING POPULATION ESTIMATES ---")
# 4. Load Population Estimates
pop_df = pd.read_excel(
    "../data_public/raw_econ/Bilingue - 2023 T1FF x 2024 CSD Population estimates.xlsx"
)
print("Population estimates loaded successfully.")


print("\n--- PROCESSING ONTARIO MONTHLY CPI INFLATION ---")
# 5. Load CPI Inflation Table
cpi_df = pd.read_csv(
    "../data_public/raw_econ/ontario_cpi_monthly.csv",
    skiprows=9,
    encoding="latin1",
)
print("CPI Inflation data loaded successfully.")


print("\n--- SAVING PROCESSED OUTPUTS ---")
# 6. Save cleaned intermediate files out to your new outputs folder
os.makedirs("../outputs", exist_ok=True)
simcoe_wp.to_excel("../outputs/simcoe_working_poverty_clean.xlsx", index=False)
print("Success! Full economic extraction pipeline completed and saved.")