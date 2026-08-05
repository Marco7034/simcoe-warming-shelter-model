import pandas as pd

wp_df = pd.read_excel(
    "../data_public/raw_econ/R25055 - Working Poor, 2023.xlsx",
    sheet_name="2023 - WP_Table_1",
    header=11,
)
print(wp_df.iloc[:, 0:2].dropna().head(30))