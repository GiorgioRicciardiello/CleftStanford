# Cleft Condition Hospital Dataset Analysis

This project focuses on processing and analyzing hospital data related to cleft conditions, including diagnoses, procedures, complications, and demographic information. It leverages datasets from 2016 and 2019 to identify cases, clean and transform data, and generate descriptive statistics and inferential insights.

## 🗂 Project Structure

- `data_construction.py`  
  Parses raw `.asc` and `.do` files, extracts relevant patient data using specified diagnosis and procedure codes, and generates a filtered dataset of cleft cases.

- `pre_process.py`  
  Cleans the filtered dataset:
  - Handles missing values
  - Maps diagnosis and procedure codes to human-readable labels
  - Derives new variables (e.g., `Prolonged_LOS`, `sex`, `race`)
  - Encodes categorical variables
  - Outputs a clean dataset ready for analysis

- `tab_one_visualization.py`  
  Generates descriptive statistics, visualizations, and contingency analyses:
  - Creates Table 1 for race-based stratification
  - Calculates complication rates by race
  - Performs chi-square tests for statistical association
  - Fits logistic regression models for binary outcomes (e.g., complications, LOS)
  - Saves result tables to CSV/Excel

## 📁 Input Files

- Raw data (`.asc`, `.do`) defined in `config/config.py`
- Cleaned datasets: 
  - `pp_data_cases.csv`
  - `pp_data_cases_clean.csv`

## 📁 Output Files

- `table1.xlsx`: Demographic table stratified by race
- `complications_by_race.csv`: Frequency of complications by race
- `race_complication_contingency_table.csv`: Chi-square contingency table
- `chi2_summary_race_complications.csv`: Chi-square test statistics
- `table_exposure_race.csv`, `table_exposure_hosp_region.csv`: Logistic regression results

## 🧰 Dependencies

- Python 3.7+
- pandas
- numpy
- scipy
- statsmodels
- tabulate
- openpyxl

Install with:

```bash
pip install -r requirements.txt
