import pandas as pd
from config.config import config, cleft_diag_codes, cleft_proc_codes, columns_categorical
from library.table_one import build_table1
from scipy.stats import chi2_contingency
from typing import List, Optional, Dict
from tabulate import tabulate
import numpy as np

if __name__ == '__main__':
    df_data = pd.read_csv(config.get('data_pre_proc_files' ).get('pp_data_cases_clean'), low_memory=True)
    output_path = config.get('results_path').get('results')

    categorical = {
        'Sex': 'sex',
        'Hospital Location': 'hosp_region',
        'Complications': 'complications',
        'Prolonged LOS > 1 day': 'prolonged_los'
    }
    ordinal = {
        'ZIPINC_QRTL (Income Quartile)': 'zipinc_qrtl'
    }
    continuous = {
        'Age at Surgery': 'age'
    }




    # %% Build table
    categorical_vars = {
        'Sex': 'sex',
        'Hospital Location': 'hosp_region',
        # 'Complications': 'complications',  # TODO: How are complications defined?
        'Prolonged LOS > 1 day': 'prolonged_los'
    }
    continuous_vars = {'Age at Surgery': 'age'}
    ordinal_vars = {'ZIPINC_QRTL (Income Quartile)': 'zipinc_qrtl'}

    df_data.loc[df_data['race'].isna(), 'race'] = 'not reported'
    df_data.loc[df_data['hosp_region'].isna(), 'hosp_region'] = 'not reported'
    # Generate Table 1
    table1 = build_table1(
        df_data.loc[~df_data['hosp_region'].isna(), :],
        'race',
        categorical_vars,
        continuous_vars,
        ordinal_vars,
        filter_col='included'
    )
    assert table1.iloc[0, :].sum() + df_data.race.isna().sum() == df_data.shape[0]
    table1.to_excel(output_path.joinpath('table1.xlsx'), index=True)
    # %% Statistical table of OR
    # # Define adjustment variables (e.g., age and sex)
    # adjust_vars = ['age', 'sex', 'zipinc_qrtl']
    # cat_adjust = ['sex', 'zipinc_qrtl']
    #
    # # Outcomes and exposures for Table 2
    # outcomes = [
    #     'has_complication_bin',
    #     'prolonged_los'
    # ]
    # # Table 2 and Table, forest plots by Race and Hospital Location
    # # Table 2: how race impacts rates of complications and prolonged LOS?
    # # Table 3: how hospital locations affect rates of complications and prolonged LOS?
    # # we want two tables not 4.
    # exposures = ['race', 'hosp_region']
    # # Table 2 and forest plots by Race
    # for outcome in outcomes:
    #     for exposure in exposures:
    #         tbl = build_or_table(df=df_data,
    #                              outcome=outcome,
    #                              exposure=exposure,
    #                              adjust_vars=adjust_vars,
    #                              cat_adjust=cat_adjust)
    #         # display_dataframe_to_user(f'OR: {outcome} by Race',tbl)
    #         # path_plot = output_path.joinpath(f'odds_rato_{outcome}_by_{exposure}.png')
    #         path_plot = None
    #         forest_plot(tbl, putput_path=path_plot, alpha=0.05, )

    #%% complications by race
    # Total number of people per race (for denominator)
    total_by_race = df_data.groupby('race').size().reset_index(name='Total Count')
    # Number of complications per race (numerator)
    complications_by_race = df_data[df_data['has_complication_bin'] == 1].groupby('race').size().reset_index(
        name='Complications')
    # Merge to get total per race and complications
    summary = pd.merge(total_by_race, complications_by_race, on='race', how='left').fillna(0)
    summary['Complications'] = summary['Complications'].astype(int)
    # Compute % within each race
    summary['Complication Rate (%)'] = (summary['Complications'] / summary['Total Count'] * 100).round(3)

    # Reorder and rename columns for clarity
    summary = summary[['race', 'Total Count', 'Complications', 'Complication Rate (%)']]
    summary.columns = ['Race', 'Total Count', 'Complications', 'Complication Rate (%)']

    # Print table
    print(tabulate(summary, headers='keys', tablefmt='psql', showindex=False))

    # Save as CSV
    summary.to_csv(output_path.joinpath('complications_by_race.csv'), index=False)

    # Chi-square test for association between race and complication
    contingency_table = pd.crosstab(df_data['race'], df_data['has_complication_bin'])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print(f"\nChi-squared test p-value: {p:.4e}")

    # Create summary DataFrame
    chi2_summary = pd.DataFrame({
        'Test Statistic': [chi2],
        'Degrees of Freedom': [dof],
        'p-value': [p]
    }).round(4)
    chi2_summary.to_csv(output_path.joinpath('chi2_summary_race_complications.csv'), index=False)

    # contigency table of the chi square
    contingency_table.columns = ['No Complication', 'Complication']
    contingency_table.index.name = 'Race'
    # Pretty print to console
    print("\nContingency Table (Race × Complication):")
    print(tabulate(contingency_table.reset_index(), headers='keys', tablefmt='psql', showindex=False))

    # Save to CSV
    contingency_table.to_csv(output_path.joinpath('race_complication_contingency_table.csv'), index=False)


    # Chi-square test for association between race and sex
    contingency_table = pd.crosstab(df_data['race'], df_data['sex'])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print(f"\nChi-squared test p-value: {p:.4e}")

    # Chi-square test for association between race and LOS
    contingency_table = pd.crosstab(df_data['race'], df_data['prolonged_los'])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print(f"\nChi-squared test p-value: {p:.4e}")


    # %% summay complications
    all_summaries = []

    # Iterate over unique races
    for race in df_data.race.unique():
        # Get value counts for complications for the current race
        summary_comp = df_data.loc[df_data['race'] == race, 'complications_lbl'].value_counts()
        # Convert to DataFrame and reset index
        summary_comp = summary_comp.reset_index()
        # Rename columns
        summary_comp.columns = ['Complications', 'Count']
        # Add a column for the race
        summary_comp['Race'] = race
        # Append to the list
        all_summaries.append(summary_comp)

    # Concatenate all DataFrames into a single DataFrame
    combined_df = pd.concat(all_summaries, ignore_index=True)

    # Reorder columns to have Race first
    combined_df = combined_df[['Race', 'Complications', 'Count']]

    # Print the combined DataFrame using tabulate
    print("Complications count for all races")
    print(tabulate(combined_df, headers='keys', tablefmt='psql', showindex=False))


    # %% statistical analysis
    var_base = ['age', 'sex', 'zipinc_qrtl']
    outcomes = ['has_complication_bin', 'prolonged_los']
    var_adjust = ['race', 'hosp_region']
    # %% Plotting
    # PLOT 1: x:ages, y:complications, continuous vs binary, hue histogram plot
    def _fit_logistic(df, outcome, exposure, covariates=None, cat_vars=None,
                      solver='bfgs', maxiter=200):
        cols = [outcome, exposure] + (covariates or [])
        clean = df.dropna(subset=cols).copy()

        # collapse rare exposure levels
        freq = clean[exposure].value_counts()
        rare = freq[freq < 10].index
        clean[exposure] = clean[exposure].replace(rare, 'Other')

        # standardize continuous covariates
        cont = [c for c in (covariates or []) if c not in (cat_vars or [])]
        if cont:
            scaler = StandardScaler()
            clean[cont] = scaler.fit_transform(clean[cont])

        # handle reference levels
        def _categorical_term(var):
            if var == "race":
                return "C(race, Treatment(reference='White'))"
            elif var == "hosp_region":
                return "C(hosp_region, Treatment(reference='Urban Teaching'))"
            elif var == exposure:
                return f"C({exposure})"  # leave as default if it's the exposure
            else:
                return f"C({var})"

        terms = [_categorical_term(exposure)] + [
            _categorical_term(c) if c in (cat_vars or []) else c for c in (covariates or [])
        ]
        formula = f"{outcome} ~ " + " + ".join(terms)

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ConvergenceWarning)
            model = smf.logit(formula, data=clean).fit(
                method=solver, maxiter=maxiter, cov_type='HC1', disp=False
            )
        return model


    def _extract_or_stats(model, prefix: str) -> pd.DataFrame:
        """Extract only the exposure‐level OR/CIs/p from a fitted model,
           and prefix the column names with e.g. 'comp_' or 'los_'. """
        params = model.params
        conf = model.conf_int()
        pvals = model.pvalues

        # keep only the terms for C(exposure)[T.level]
        mask = params.index.str.startswith('C(')
        df = pd.DataFrame({
            prefix + 'OR': np.exp(params[mask]),
            prefix + 'LCL': np.exp(conf.loc[mask, 0]),
            prefix + 'UCL': np.exp(conf.loc[mask, 1]),
            prefix + 'p': pvals[mask]
        })
        df.index = df.index.str.replace(r'.*T\.(.*)\]', r'\1', regex=True)
        return df.round({prefix + 'OR': 2, prefix + 'LCL': 3, prefix + 'UCL': 3, prefix + 'p': 3})


    def build_combined_or_table(df: pd.DataFrame,
                                exposure: str,
                                outcomes: List[str],
                                adjust_vars: Optional[List[str]] = None,
                                cat_adjust: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Fits an unadjusted logistic regression for each outcome (outcome ~ exposure).

        Extracts odds ratio (OR), 95% CI, and p-value for each level of the exposure.

        Fits an adjusted logistic regression (outcome ~ exposure + age + sex + zipinc_qrtl),
        again extracting ORs/CIs/p.

        Merges the two outcomes side by side, and stacks the unadjusted + adjusted results into one table that has,
        for each exposure level, both complication and LOS ORs under both model types.

        Returns a single DataFrame with columns for each outcome:
           OR_comp, LCL_comp, UCL_comp, p_comp,
           OR_los,  LCL_los,  UCL_los,  p_los
        for both Unadjusted and Adjusted models, and rows = exposure levels.
        """
        blocks = []
        for model_type, covs in [('Unadjusted', None), ('Adjusted', adjust_vars)]:
            # for each outcome, get its little stats‐df
            dfs = []
            for Y in outcomes:
                prefix = 'comp_' if 'complication' in Y else 'los_'
                m = _fit_logistic(df, Y, exposure, covariates=covs, cat_vars=cat_adjust)
                stats = _extract_or_stats(m, prefix)
                dfs.append(stats)
            # merge the two outcome‐stats on the index (exposure levels)
            merged = reduce(lambda a, b: a.join(b), dfs)
            merged['Model'] = model_type
            blocks.append(merged.reset_index().rename(columns={'index': exposure}))
        # stack Unadj + Adj
        result = pd.concat(blocks, ignore_index=True)
        # reorder columns
        cols = [exposure, 'Model'] + \
               [c for c in result.columns if c not in {exposure, 'Model'}]
        return result[cols]


    # --- usage in main script ---
    adjust_vars = ['age', 'sex', 'zipinc_qrtl']
    cat_adjust = ['sex', 'zipinc_qrtl']
    outcomes = ['has_complication_bin', 'prolonged_los']
    exposures = ['race', 'hosp_region']

    tables = {}
    for exp in exposures:
        tables[exp] = build_combined_or_table(df_data,
                                              exp,
                                              outcomes,
                                              adjust_vars,
                                              cat_adjust)
        tables[exp].to_csv(output_path.joinpath(f'table_exposure_{exp}.csv'), index=False)

    # %%

