"""
pre_process.py
--------------
This script loads the merged KID dataset, cleans the data, recodes missing (negative) values
to NaNs, drops columns that are all missing, and creates derived variables (such as Prolonged LOS, Sex, Race, and Inclusion flags).
Additional processing steps are proposed to make the dataset suitable for analysis.
"""
import pandas as pd
from config.config import (config,
                           cleft_diag_codes,
                           cleft_proc_codes,
                           columns_categorical,
                           complications,
                           complications_procedures,
                           mapper)
import numpy as np
from typing import List, Dict, Tuple
from collections import Counter

def map_complications(
        df: pd.DataFrame,
        diag_cols: List[str],
        proc_cols: List[str],
        comp_diag: Dict[str, str],
        comp_proc: Dict[str, str]
) -> pd.DataFrame:
    """
    Efficiently maps complications to binary and labeled columns in a large DataFrame using vectorized boolean masks.

    - Creates a binary column `complications_bin` which is 1 if any complication code is found in diagnosis or procedure columns.
    - Creates a `complications_labels` column containing a comma-separated string of the matched complication labels.

    Parameters:
    - df (pd.DataFrame): The input dataframe with diagnosis and procedure codes.
    - diag_cols (List[str]): Columns containing diagnosis codes.
    - proc_cols (List[str]): Columns containing procedure codes.
    - comp_diag (Dict[str, str]): Diagnosis complication code-to-label mapping.
    - comp_proc (Dict[str, str]): Procedure complication code-to-label mapping.

    Returns:
    - pd.DataFrame: Original DataFrame with two new columns: `complications_bin` and `complications_labels`.
    """



    # Initialize containers
    diag_labels = pd.DataFrame('', index=df.index, columns=['labels_diag'])
    proc_labels = pd.DataFrame('', index=df.index, columns=['labels_proc'])

    # Vectorized matching for diagnosis columns
    for code, label in comp_diag.items():
        match_mask = df[diag_cols].isin([code]).any(axis=1)
        diag_labels.loc[match_mask, 'labels_diag'] += label + ', '

    # Vectorized matching for procedure columns
    for code, label in comp_proc.items():
        match_mask = df[proc_cols].isin([code]).any(axis=1)
        proc_labels.loc[match_mask, 'labels_proc'] += label + ', '

    # Combine and clean
    labels_combined = (
            diag_labels['labels_diag'].fillna('') +
            proc_labels['labels_proc'].fillna('')
    ).str.strip(', ')

    df['complications_labels'] = labels_combined
    df['complications_bin'] = (labels_combined != '').astype(int)

    return df
if __name__ == '__main__':
    # df_data = pd.read_csv(config['data_pre_proc_files']['merged'], low_memory=False) # nrows=3000)
    # use only cases
    df_data = pd.read_csv(config.get('data_pre_proc_files').get('pp_data_cases'), low_memory=False) # nrows=3000)


    # # lower caps all columns
    # df_data.columns = map(str.lower, df_data.columns)
    print(f'Initial dataset dimensions: {df_data.shape}')

    # === Missing value recoding ===
    # For many HCUP/KID variables, negative values indicate missing. For numeric columns, we replace any negative value with NaN.
    num_cols = df_data.select_dtypes(include=[np.number]).columns
    df_data[num_cols] = df_data[num_cols].mask(df_data[num_cols] < 0, np.nan)

    diag_cols = [col for col in df_data.columns if col.startswith("I10_DX")]
    proc_cols = [col for col in df_data.columns if col.startswith("I10_PR")]

    # %% Complications
    # complication_codes = set(complications.keys())
    # complications_procedures_codes = set(complications_procedures.keys())
    #
    # mask_complications_diag_cols = df_data[diag_cols].isin(complication_codes).any(axis=1)
    # mask_complications_proc_cols = df_data[proc_cols].isin(complications_procedures_codes).any(axis=1)
    # # If any has a complication return True, else False
    # mask_complications_any = mask_complications_diag_cols | mask_complications_proc_cols
    # df_data['has_complication_bin'] = mask_complications_any.astype(int)

    # To record the matched complication(s) we do the following:
    # Step 1: Initialize a column with empty strings
    df_data['complications_lbl'] = ''

    # Step 2: Match diagnosis complications
    for code, label in complications.items():
        mask = df_data[diag_cols].isin([code]).any(axis=1)
        df_data.loc[mask, 'complications_lbl'] += label + ', '

    # Step 3: Match procedure complications (e.g., transfusions)
    for code, label in complications_procedures.items():
        mask = df_data[proc_cols].isin([code]).any(axis=1)
        df_data.loc[mask, 'complications_lbl'] += label + ', '

    # Step 4: Clean up trailing commas and spaces
    df_data['complications_lbl'] = df_data['complications_lbl'].str.strip(', ')

    # Binary flag (already added earlier)
    df_data['has_complication_bin'] = (df_data['complications_lbl'] != '').astype(int)

    # %% === Boolean flag for diagnosis and procedure ===
    # Create boolean flag for cleft palate diagnosis codes in any diagnosis column.
    # Create boolean masks (vectorized, much faster)
    diag_codes = [*cleft_diag_codes.keys()]
    proc_codes = [*cleft_proc_codes.keys()]
    mask_diag = df_data[diag_cols].isin(diag_codes).any(axis=1)
    mask_proc = df_data[proc_cols].isin(proc_codes).any(axis=1)

    df_data['has_diag'] = df_data[diag_cols].apply(
        lambda row: any(str(code).strip() in cleft_diag_codes for code in row.dropna().astype(str)), axis=1)
    # Similarly, flag for procedure codes.
    df_data['has_proc'] = df_data[proc_cols].apply(
        lambda row: any(str(code).strip() in cleft_proc_codes for code in row.dropna().astype(str)), axis=1)



    #%% === Boolean flag and matched codes for diagnosis and procedures ===

    # Diagnosis and procedure code lists
    diag_codes = list(cleft_diag_codes.keys())
    proc_codes = list(cleft_proc_codes.keys())

    # Diagnosis mask and matched codes
    df_data['diag_match_codes'] = df_data[diag_cols].apply(
        lambda row: ', '.join([code for code in row.dropna().astype(str) if code in cleft_diag_codes]),
        axis=1
    )
    df_data['has_diag'] = df_data['diag_match_codes'].str.len() > 0

    # Procedure mask and matched codes
    df_data['proc_match_codes'] = df_data[proc_cols].apply(
        lambda row: ', '.join([code for code in row.dropna().astype(str) if code in cleft_proc_codes]),
        axis=1
    )
    df_data['has_proc'] = df_data['proc_match_codes'].str.len() > 0

    # Count diagnosis occurrences
    df_data['diag_match_codes'].value_counts()
    # Count procedures occurrences
    df_data['proc_match_codes'].value_counts()

    # === Drop diagnosis and procedures columns ===
    col_drop = diag_cols + proc_cols
    df_data.drop(columns=col_drop, inplace=True)
    print(f'After dropping diagnosis and procedure columns: \n\t\t{df_data.shape}')

    # === Drop columns that are all missing ===
    df_data.dropna(axis=1, how="all", inplace=True)
    print(f'After dropping all nan columns: \n\t\t{df_data.shape}')

    # %% === Derive New Variables ===
    # Create "Prolonged_LOS": flag patients with LOS > 1 day.
    days_prolonged_stay = 1
    if 'LOS' in df_data.columns:
        df_data['Prolonged_LOS'] = (df_data['LOS'] > days_prolonged_stay).astype(int)
        # Bin LOS into [0–1], [2–4], [5+]
        bins = [0, 1, 4, np.inf]
        labels = ['0–1 days', '2–4 days', '5+ days']
        df_data['LOS_group'] = pd.cut(
            df_data['LOS'],
            bins=bins,
            labels=labels,
            right=True,  # intervals are (a, b]
            include_lowest=True  # include 0 in the first bin
        )
    else:
        print("Warning: 'LOS' column not found in the dataset.")

    # Create "Sex" variable derived from the FEMALE indicator.
    # We assume in the dataset: FEMALE == 1 means female, FEMALE == 0 (or otherwise) means male.
    if 'FEMALE' in df_data.columns:
        df_data['sex'] = df_data['FEMALE'].apply(lambda x: "Female" if x == 1 else "Male")
    else:
        print("Warning: 'FEMALE' column not found in the dataset.")
    df_data.drop(columns=['FEMALE'], inplace=True)

    # %% === Re-map cateogircal columns
    # 3. Recode Race: Using typical KID codew
    if 'RACE' in df_data.columns:
        df_data['RACE'] = df_data['RACE'].map(mapper.get('race'))
        df_data['RACE'] = df_data['RACE'].astype('category')
        # df_data['RACE'].replace({''})
    else:
        print("Warning: 'RACE' column not found in the dataset.")

    # # MISSING MAPPINGS
    df_data['HOSP_REGION'] = df_data['HOSP_REGION'].map(mapper.get('hosp_region'))
    df_data['HOSP_REGION'] = df_data['HOSP_REGION'].astype('category')


    # convert to categorical type
    for col in columns_categorical:
        if col in df_data.columns:
            df_data[col] = df_data[col].astype('category')

    #%% === Standardize column names (e.g., lower-case, remove whitespace).
    df_data.columns = df_data.columns.str.strip().str.lower()
    df_data.drop_duplicates(inplace=True)

    # %% === Data Transformations
    df_data['totchg_log1p'] = np.log1p(df_data['totchg'])
    df_data['age_z'] = (df_data['age'] - df_data['age'].mean()) / df_data['age'].std()

    df_data['discwt'] = df_data['discwt'].round(3)

    # $$ === ID columns rename
    col_ids = {'hosp_kid':'id_hosp_kid',
               'recnum':'id_recnum', }
    df_data.rename(columns=col_ids, inplace=True)

    # %% === Sort columns
    id_cols = sorted([col for col in df_data.columns if col.startswith("id_")])
    # Head columns
    special_cols = [col for col in ['age', 'sex', 'race', 'died'] if col in df_data.columns]
    # remaining columns
    remaining_cols = sorted([col for col in df_data.columns if col not in id_cols + special_cols])
    new_order = id_cols + special_cols + remaining_cols
    df_data = df_data[new_order]

    # === Save the Pre-Processed Data ===
    # save only cases dataset
    df_data.to_csv(config.get('data_pre_proc_files').get('pp_data_cases_clean'), index=False)
    print(f'Pre-process only cases dataset saved')


