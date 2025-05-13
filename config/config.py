"""
Configuration file of the project
"""
import pathlib

# Define root path
root_path = pathlib.Path(__file__).resolve().parents[1]

# Define raw data path and individual raw data files
data_raw_path = root_path.joinpath('data', 'raw_data')
data_raw_path_2016 = data_raw_path.joinpath('KID_2016_Core.ASC')
data_raw_path_2019 = data_raw_path.joinpath('KID_2019_Core.ASC')

# Define pre-processed data path and individual pre-processed data files
data_pre_proc_path = root_path.joinpath('data', 'pp_data')
data_pp_path_2016 = data_pre_proc_path.joinpath('KID_2016_Core.csv')
data_pp_path_2019 = data_pre_proc_path.joinpath('KID_2019_Core.csv')
data_pp_path_filtered_patients_2016 = data_pre_proc_path.joinpath('KID_2016_Core_Filtered.csv')
data_pp_path_filtered_patients_2019 = data_pre_proc_path.joinpath('KID_2019_Core_Filtered.csv')
data_pp_path_merged = data_pre_proc_path.joinpath('kid_merged.csv')
data_pp_path = data_pre_proc_path.joinpath('pp_data.csv')
data_pp_path_cases = data_pre_proc_path.joinpath('pp_data_cases.csv')
data_pp_path_cases_clean = data_pre_proc_path.joinpath('pp_data_cases_clean.csv')
do_pp_path_2016 = data_pre_proc_path.joinpath('df_do_2016.csv')
do_pp_path_2019 = data_pre_proc_path.joinpath('df_do_2019.csv')

# Documentation file of the columns
docs_path = root_path.joinpath('docs')
docs_2016 = docs_path.joinpath('StataLoad_KID_2016_Core.Do')
docs_2019 = docs_path.joinpath('StataLoad_KID_2019_Core.Do')


# Define results path
results_path = root_path.joinpath('results')
# Configuration dictionary
config = {
    'root_path': root_path,
    'data_raw_path': data_raw_path,
    'data_pre_proc_path': data_pre_proc_path,
    # 'results_path': results_path,
    'data_raw_files': {
        '2016': data_raw_path_2016,
        '2019': data_raw_path_2019,

    },
    'data_pre_proc_files': {
        '2016': data_pp_path_2016,
        '2019': data_pp_path_2019,
        'filtered_patients_2016': data_pp_path_filtered_patients_2016,
        'filtered_patients_2019': data_pp_path_filtered_patients_2019,
        'merged': data_pp_path_merged,
        'pp_data': data_pp_path,
        'pp_data_cases': data_pp_path_cases,
        'pp_data_cases_clean': data_pp_path_cases_clean,
        'do_frame_2016': do_pp_path_2016,
        'do_frame_2019': do_pp_path_2019,

    },

    'doc_files': {
        '2016': docs_2016,
        '2019': docs_2019,
    },

    'results_path': {
        'results': results_path,
    },

}


cleft_diag_codes = {
    'Q351': 'Cleft Palate',
    'Q353': 'Cleft Palate',
    'Q355': 'Cleft Palate',
    'Q357': 'Cleft Palate',
    'Q359': 'Cleft Palate',
    'Q370': 'Cleft Palate',
    'Q371': 'Cleft Palate',
    'Q372': 'Cleft Palate',
    'Q373': 'Cleft Palate',
    'Q374': 'Cleft Palate',
    'Q375': 'Cleft Palate',
    'Q378': 'Cleft Palate',
    'Q37p': 'Cleft Palate',
}

cleft_proc_codes = {
    "0CQ30ZZ": "Repair Soft Palate, Open Approach",
    "0CQ33ZZ": "Repair Soft Palate, Percutaneous Approach",
    "0CQ3XZZ": "Repair Soft Palate, External Approach",
    "0CQ20ZZ": "Repair Hard Palate, Open Approach",
    "0CQ23ZZ": "Repair Hard Palate, Percutaneous Approach",
    "0CQ2XZZ": "Repair Hard Palate, External Approach"

}

# All complications BESIDES Transfusion can be found in the diagnosis columns
# Transfusion is found in the procedures column.
complications = {
    "J9601": "Airway/Respiratory Failure",
    "J9602": "Airway/Respiratory Failure",
    "J9691": "Airway/Respiratory Failure",
    "L7622": "Hemorrhage, Hematoma",
    "M96811": "Hemorrhage, Hematoma",
    "M96831": "Hemorrhage, Hematoma",
    "J13": "Pneumonia",
    "J181": "Pneumonia",
    "J151": "Pneumonia",
    "J154": "Pneumonia",
    "B9683": "Pneumonia",
    "J159": "Pneumonia",
    "J168": "Pneumonia",
    "J158": "Pneumonia",
    "J188": "Pneumonia",
    "J189": "Pneumonia",
    "J95851": "Pneumonia",
    "J9811": "Pneumonia",
    "J9819": "Pneumonia",
    "K6811": "Post-Operative Infection",
    "M8618": "Post-Operative Infection",
    "M86.28": "Post-Operative Infection",
    "M8619": "Post-Operative Infection",
    "M8629": "Post-Operative Infection",
    "M4630": "Post-Operative Infection",
    "M869": "Post-Operative Infection",
    "K122": "Post-Operative Infection",
    "L03211": "Post-Operative Infection",
    "L03212": "Post-Operative Infection",
    "L03213": "Post-Operative Infection",
    "T8130XA": "Wound Disruption",
    "T8131XA": "Wound Disruption",
}

complications_procedures = {
    "30233H0": "Transfusion",
    "30233N0": "Transfusion",
    "30243H0": "Transfusion",
    "30243N0": "Transfusion",
    "30233H1": "Transfusion",
    "30243H1": "Transfusion",
    "30233N1": "Transfusion",
    "30233P1": "Transfusion",
    "30243N1": "Transfusion",
    "30243P1": "Transfusion",
    "30233R1": "Transfusion",
    "30243R1": "Transfusion",
    "30233T1": "Transfusion",
    "30233V1": "Transfusion",
    "30233W1": "Transfusion",
    "30243T1": "Transfusion",
    "30243V1": "Transfusion",
    "30243W1": "Transfusion",
    "30233J1": "Transfusion",
    "30233K1": "Transfusion",
    "30233L1": "Transfusion",
    "30233M1": "Transfusion",
    "30243J1": "Transfusion",
    "30243K1": "Transfusion",
    "30243L1": "Transfusion",
    "30243M1": "Transfusion"
}
complications_all = {**complications, **complications_procedures}

columns_categorical = [
    'sex',
    'race',
    "HOSP_KID",         # KID hospital number; may be used for grouping
    "AMONTH",           # Admission month
    "AWEEKEND",         # Admission day is a weekend indicator
    "DIED",             # Died during hospitalization (yes/no)
    "DISPUNIFORM",      # Disposition of patient (uniform)
    "DQTR",             # Discharge quarter
    "DRG",              # Diagnosis-Related Group in effect on discharge date
    "DRGVER",           # DRG grouper version
    "DRG_NoPOA",        # DRG calculated without POA
    "DXVER",            # Diagnosis Version
    "ELECTIVE",         # Elective versus non-elective admission indicator
    "FEMALE",           # Indicator of sex (0/1)
    "HCUP_ED",          # HCUP Emergency Department service indicator
    "HOSP_REGION",      # Region of hospital
    "I10_HOSPBRTH",     # Indicator of birth in this hospital
    "I10_UNCBRTH",      # Normal uncomplicated birth indicator
    "KID_STRATUM",      # Stratum used to sample hospital
    "MDC",              # Major Diagnostic Category in effect on discharge date
    "MDC_NoPOA",        # MDC calculated without POA
    "PAY1",             # Primary expected payer (uniform)
    "PL_NCHS",          # Patient Location: NCHS Urban-Rural Code
    "PRVER",            # Procedure Version
    "RACE",             # Race (uniform)
    "TRAN_IN",          # Transfer in indicator
    "TRAN_OUT",         # Transfer out indicator
    "ZIPINC_QRTL",      # Median household income quartile for patient ZIP Code
    "I10_BIRTH",        # ICD-10-CM Birth Indicator
    "I10_DELIVERY",     # ICD-10-CM Delivery Indicator
    "PCLASS_ORPROC"     # Indicates operating room procedure on the record
]

mapper = {
    'race': {
        1: 'White',
        2: 'Black',
        3: 'Hispanic',
        4: 'Asian of Pacific Islander',
        5: 'Other',  # native american parsed as other
        6: 'Other',
    },

    'hosp_region': {
        1: 'Rural',
        2: 'Urban Nonteaching',
        3: 'Urban Teaching',
    },

    'zipinc_qrtl': {
        1 : '0-25th percentile',
        2 : '26-50th percentile',
        3 : '51-75th percentile',
        4 : '76-100th percentile'
    },

}