import pandas as pd
import numpy as np
import os

print("--- GENERATING DUMMY SHELTER DATA ---")

# Ensure the internal data folder exists (though gitignored)
os.makedirs("../data_internal", exist_ok=True)

# 12 distinct fake shelters
shelters = [
    "Barrie_Warming_Center", "Orillia_Lighthouse", "Midland_Shelter_Hope",
    "Collingwood_Emergency", "Alliston_Outreach", "Bradford_Winter_Haven",
    "Innisfil_Safe_Beds", "Wasaga_Beach_Support", "Penetanguishene_Relief",
    "Springwater_Sanctuary", "Tay_Township_Refuge", "Essa_Emergency_Cots"
]

# Winter months spanning Nov 2023 to Apr 2026
months = [
    "2023-11", "2023-12", "2024-01", "2024-02", "2024-03", "2024-04",
    "2024-11", "2024-12", "2025-01", "2025-02", "2025-03", "2025-04",
    "2025-11", "2025-12", "2026-01", "2026-02", "2026-03", "2026-04"
]

# Create exactly 79 rows of fake data to match Copilot's output
np.random.seed(42) # For reproducibility
data = {
    "ShelterName_Clean": np.random.choice(shelters, 79),
    "YearMonth": np.random.choice(months, 79),
    "TotalAdmissions": np.random.randint(10, 150, 79), # Fake admission counts
    "TotalDischarges": np.random.randint(8, 145, 79),
    "ZeroDayAdmissions": np.random.randint(0, 20, 79)
}

dummy_df = pd.DataFrame(data)

# Sort it to look like a clean chronological panel
dummy_df = dummy_df.sort_values(by=["ShelterName_Clean", "YearMonth"])

# Save directly to the gitignored internal folder
dummy_path = "../data_internal/MODEL_TRAINING_PANEL.csv"
dummy_df.to_csv(dummy_path, index=False)

print(f"Success! Generated 79 rows of fake data at {dummy_path}")