class DiagnosisSelector:
    def __init__(self, df):
        self.df = df

    def filter_by_diagnoses(self,
                            diagnosis_codes,
                            diagnosis_prefix='I10_DX',
                            procedure_prefix='I10_PR'):
        diag_cols = [col for col in self.df.columns if col.startswith(diagnosis_prefix)]
        proc_cols = [col for col in self.df.columns if col.startswith(procedure_prefix)]
        relevant_cols = diag_cols + proc_cols

        mask = self.df[relevant_cols].apply(
            lambda row: any(code in str(row[col]) for col in relevant_cols for code in diagnosis_codes), axis=1
        )
        return self.df[mask]
