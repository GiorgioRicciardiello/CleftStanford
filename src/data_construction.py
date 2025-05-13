"""
Main script to parse the dataset to csv, parse the .Do do a csv and filtered the frame by the patients we want given
the diagnosis and procedures codes
"""
from library.asc_parser import ASCParser
from library.diagnosis_selector import DiagnosisSelector
from config.config import config, cleft_diag_codes, cleft_proc_codes, complications, complications_procedures
import pandas as pd


for raw_year_date in ['2016', '2019']:
    do_file_path = config['doc_files'][raw_year_date]
    asc_file_path = config['data_raw_files'][raw_year_date]
    asc_csv_path = config['data_pre_proc_files'][raw_year_date]
    do_csv_path = config['data_pre_proc_files'][f'do_frame_{raw_year_date}']
    if asc_csv_path.exists():
        continue
    # construct parser
    parser_do_file = ASCParser(
        do_file_path=do_file_path,
        do_file_output_path=do_csv_path,        # ahora solo para labels
        asc_file_path=asc_file_path,
        asc_file_output_path=asc_csv_path       # ahora solo para datos
    )

    # Execute parsing
    parser_do_file.parse_do_file()
    df_asc = parser_do_file.save_asc_to_csv()  # CSV con datos reales
    df_do = parser_do_file.save_do_as_csv()    # CSV con diccionario

    print(f"[✓] Year {raw_year_date} Completed:")
    print(f"     - ASC CSV: {asc_csv_path}")
    print(f"     - DO Labels CSV: {do_csv_path}")

# Filter the dataset into the patients we want
df_include_cases = pd.DataFrame()
for raw_year_date in ['2016', '2019']:
    #%% Input
    df_data = pd.read_csv(config['data_pre_proc_files'][raw_year_date], low_memory=False)
    df_do_csv = pd.read_csv(config['data_pre_proc_files'][f'do_frame_{raw_year_date}'])

    #%% Output
    path_filtered = config['data_pre_proc_files'][f'filtered_patients_{raw_year_date}']

    # if path_filtered.exists():
    #     continue
    # Diagnosis and procedure columns

    diag_cols = [col for col in df_data.columns if col.startswith("I10_DX")]
    proc_cols = [col for col in df_data.columns if col.startswith("I10_PR")]

    diag_codes = set(cleft_diag_codes.keys())
    proc_codes = set(cleft_proc_codes.keys())

    # Create boolean masks (vectorized, much faster), checks for exact matches
    mask_diag = df_data[diag_cols].isin(diag_codes).any(axis=1)
    mask_proc = df_data[proc_cols].isin(proc_codes).any(axis=1)

    # Debug Check
    # Create a dictionary where each key is a column name and its value is a Series of unique non-NaN values
    # unique_data_diag_codes = {col: pd.Series(df_data[col].dropna().unique()) for col in diag_cols}
    # df_unique_diag = pd.DataFrame(unique_data_diag_codes)
    #
    # unique_data_diag_proc = {col: pd.Series(df_data[col].dropna().unique()) for col in proc_cols}
    # df_unique_proc = pd.DataFrame(unique_data_diag_proc)

    # %% Inclusion flag
    # OLD INCLUSION: 1 if Diagnosis OR Procedure Code Exist
    # df_data['included'] = (mask_diag | mask_proc).astype(int)
    # Current INCLUSION: 1 if Diagnosis AND Procedure Code Exist
    df_data['included'] = (mask_diag & mask_proc).astype(int)

    # Count alternative inclusion logic
    included_diag = (mask_diag).sum()
    includes_proc = (mask_proc).sum()
    included_diag_or_proc = (mask_diag | mask_proc).sum()
    included_diag_and_proc = (mask_diag & mask_proc).sum()

    print(f'===== {raw_year_date} ====')
    print(f'Full data: {df_data.shape[0]}')
    print(f"Subjects with Diagnosis ({raw_year_date}): {included_diag}")
    print(f"Subjects with Procedures ({raw_year_date}): {includes_proc}")
    print(f"Subjects with Diagnosis OR Procedure ({raw_year_date}) (NOT USED): {included_diag_or_proc}")
    print(f"Subjects with Diagnosis AND Procedure ({raw_year_date}): {included_diag_and_proc}")

    # ===== 2016 ====
    # Full data: 3117413
    # Subjects with Diagnosis (2016): 7845
    # Subjects with Procedures (2016): 1136
    # Subjects with Diagnosis OR Procedure (2016) (NOT USED): 8134
    # Subjects with Diagnosis AND Procedure (2016): 847
    # Subject Included in the analysis 2016: 847
    # ===== 2019 ====
    # Full data: 3089283
    # Subjects with Diagnosis (2019): 6805
    # Subjects with Procedures (2019): 845
    # Subjects with Diagnosis OR Procedure (2019) (NOT USED): 6995
    # Subjects with Diagnosis AND Procedure (2019): 655
    # %%
    # year columns
    df_data['year'] = int(raw_year_date)

    df_data.to_csv(path_filtered, index=False)

    print(f"Subject Included in the analysis {raw_year_date}: {df_data['included'].sum()}")
    df_cases = df_data.loc[df_data['included'] == 1, :]
    df_include_cases = pd.concat([df_include_cases, df_cases], axis=0)

# Total patients for year 2016: 3117413
# Inclusion flagged patients for year 2016: 647

# Total patients for year 2019: 3089283
# Inclusion flagged patients for year 2019: 517

df_include_cases.to_csv(config.get('data_pre_proc_files').get('pp_data_cases'), index=False)
print(f'Merged only cases dataset saved: {df_include_cases.shape}')

# # Merge into a final dataset
# df_data_merged = pd.DataFrame()
# for raw_year_date in ['2016', '2019']:
#     path_filtered = config['data_pre_proc_files'][f'filtered_patients_{raw_year_date}']
#     df_filtered = pd.read_csv(path_filtered, low_memory=False)
#     print(f"Processing year {raw_year_date}: {df_filtered.shape[0]} records loaded from {path_filtered}")
#     df_data_merged = pd.concat([df_data_merged, df_filtered])
#
# print(f"Final merged dataset contains {df_data_merged.shape[0]} records from 2016 and 2019.")
# df_data_merged.sort_values(by=['included'], ascending=False, inplace=True)

# %% Save file
# df_data_merged.to_csv(config['data_pre_proc_files']['merged'], index=False)
# print(f'Pre-process full dataset saved: {df_data_merged.shape}')
# # save only cases
# df_cases = df_data.loc[df_data['included'] == 1, :]
# df_cases.to_csv(config.get('data_pre_proc_files').get('pp_data_cases'), index=False)
# print(f'Pre-process only cases dataset saved: {df_cases.shape}')
# Final merged dataset contains 6206696 records from 2016 and 2019.
# cases dataset 1502





