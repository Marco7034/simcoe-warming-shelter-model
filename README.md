# Municipal Warming Shelter Capacity Hazard & Surge Forecasting

## Project Overview
This project was developed for the County of Simcoe to transition municipal winter warming shelter management from retrospective reporting to proactive predictive analytics. 

By modeling the intersection of short-term meteorological extremes and medium-term macroeconomic housing pressures, this end-to-end data pipeline estimates the probability of capacity surge events across the shelter system, allowing municipal staff to proactively allocate resources, approve staffing, and open overflow sites before crises occur.

*Note: To comply with municipal privacy and security standards, the internal shelter utilization dataset included in this repository (`MODEL_TRAINING_PANEL.csv`) contains fully synthetic/dummy data. The pipeline and statistical architecture remain fully functional.*

## Technical Stack & Methodologies
* **Programming:** Python (`pandas`, `numpy`, `statsmodels`)
* **Visualization:** Power BI, DAX, Power Query
* **Statistical Framework:** Discrete-Time Logistic Hazard Modeling (Survival Analysis), Maximum Likelihood Estimation (MLE), Time-Series Feature Engineering

## Data Architecture
1. **Internal Shelter Panel:** Longitudinal shelter admission records engineered to create dynamic, shelter-specific 85th-percentile capacity thresholds (Target Variable: `Admission_Surge_Flag`).
2. **Meteorological Shocks:** High-frequency daily weather observations from Environment Canada, aggregated into monthly hazard indicators (e.g., `Extreme_Min_Temp`).
3. **Macroeconomic Indicators:** Monthly Ontario Consumer Price Index (CPI) and localized 2-bedroom rental market rates from the CMHC, engineered with 3-month predictive lags to model delayed financial pressure on vulnerable populations.

## Key Statistical Findings
The discrete-time logistic hazard model successfully converged, demonstrating strong explanatory power for complex capacity systems:
* **Meteorological Trigger:** A 5°C drop in extreme minimum temperature elevates the odds of a facility surge by **~28.7%** ($\beta = -0.0505$).
* **Housing Market Pressure:** Lagged average local rent exhibited a positive marginal relationship ($\beta = 0.0112$), validating that escalating localized rental costs create a delayed upward shift in shelter baseline demand.
* *(Note: Multicollinearity between CPI and Rent was actively managed during model specification to prevent singular matrix failures).*

## How to Run the Pipeline
Ensure you have the required dependencies installed (`pandas`, `numpy`, `statsmodels`). Navigate to the `/src` directory and execute the pipeline in the following order:

```bash
# 1. Clean shelter data and engineer the 85th-percentile surge flags
python prep_shelter_data.py

# 2. Compress daily weather and lag economic data into the master design matrix
python build_shock_matrix.py

# 3. Fit the logistic hazard model and export the scored probability forecast
python train_hazard_model.py