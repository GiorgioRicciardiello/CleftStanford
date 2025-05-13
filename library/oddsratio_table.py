import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from typing import List, Optional, Dict, Tuple
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import pathlib

# === EffectMeasurePlot class with hue by model ===
class EffectMeasurePlot:
    def __init__(self, label: List[str],
                 effect_measure: List[float],
                 lcl: List[float],
                 ucl: List[float],
                 p_value: List[float],
                 model: List[str],
                 alpha: float = 0.05,
                 decimal_effect: Optional[int] = 2,
                 decimal_ci: Optional[int] = 3,
                 decimal_pval: Optional[int] = 6,
                 ):
        self.df = pd.DataFrame({
            'study': label,
            'OR': np.round(effect_measure, decimal_effect),
            'LCL': np.round(lcl, decimal_ci),
            'UCL': np.round(ucl, decimal_ci),
            'pvalue': np.round(p_value, decimal_pval),
            'Model': model
        })
        self.df['OR2'] = self.df['OR']
        self.df['LCL_dif'] = np.abs(self.df['OR2'] - self.df['LCL'])
        self.df['UCL_dif'] = np.abs(self.df['UCL'] - self.df['OR2'])
        self.alpha = alpha
        self.ci = f'{np.round((1 - alpha) * 100, 2)}% CI'
        self.scale = 'linear'
        self.center = 1
        self.model_colors = {'Unadjusted': 'tab:blue', 'Adjusted': 'tab:orange'}
        self.errc = 'dimgrey'
        self.shape = 'o'
        self.linec = 'gray'

    def set_labels(self, **kwargs):
        self.ci = kwargs.get('conf_int', self.ci)
        self.scale = kwargs.get('scale', self.scale)
        self.center = kwargs.get('center', self.center)

    def set_colors(self, **kwargs):
        self.model_colors.update(kwargs.get('model_colors', {}))
        self.linec = kwargs.get('linecolor', self.linec)

    @staticmethod
    def _check_statistical_significance(odd_ratio: float,
                                        lcl: float,
                                        ucl: float,
                                        pvalue: float,
                                        alpha: float) -> bool:
        return (pvalue < alpha) and not (lcl <= 1 <= ucl)

    def plot(
            self,
            figsize: Tuple[int, int] = (12, 6),
            size: int = 5,
            text_size: int = 10,
            path_save: pathlib.Path = None
    ):
        """
        Draw a forest plot with side-by-side table, excluding the intercept and
        ordering table rows to match the plot y-axis.
        """
        # Exclude intercept and get ordered parameters
        params = [p for p in self.df['study'].unique() if p != 'Intercept']
        base_y = {param: i for i, param in enumerate(params)}
        height = len(params)

        # Compute x-axis limits over non-intercept
        df_non = self.df[self.df['study'].isin(params)]
        lo, hi = df_non['LCL'].min(), df_non['UCL'].max()
        pad = (hi - lo) * 0.1
        mini, maxi = lo - pad, hi + pad

        # Figure and GridSpec
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        gs = gridspec.GridSpec(1, 2, figure=fig,
                               width_ratios=[3, 2], wspace=0.05)
        ax = fig.add_subplot(gs[0, 0])
        tbl_ax = fig.add_subplot(gs[0, 1])

        # Reference line
        if self.scale == 'log':
            ax.set_xscale('log')
        ax.axvline(self.center, linestyle='--', color=self.linec, alpha=0.7)

        # Plot points, skipping intercept
        for _, row in self.df.iterrows():
            if row['study'] == 'Intercept':
                continue
            offset = 0.2 if row['Model'] == 'Adjusted' else -0.2
            y = base_y[row['study']] + offset
            sig = self._check_statistical_significance(
                row['OR2'], row['LCL'], row['UCL'], row['pvalue'], self.alpha
            )
            mshape = self.shape if sig else 'X'
            mcolor = self.model_colors.get(row['Model'], 'black')
            ax.errorbar(
                x=row['OR2'], y=y,
                xerr=[[row['LCL_dif']], [row['UCL_dif']]],
                fmt='none', ecolor=mcolor, elinewidth=1
            )
            ax.scatter(
                x=row['OR2'], y=y,
                marker=mshape, s=size * 20,
                color=mcolor, edgecolors='black'
            )

        # Y-axis
        yticks = [base_y[p] for p in params]
        ax.set_yticks(yticks)
        ax.set_yticklabels(params)
        ax.set_ylim(-1, height)
        ax.set_xlim(mini, maxi)
        ax.set_xlabel('Odds Ratio')
        ax.grid(axis='x', linestyle='--', alpha=0.5)

        # Legend
        handles = []
        for m, col in self.model_colors.items():
            handles.append(mlines.Line2D(
                [], [], color=col, marker='o', linestyle='None',
                markersize=6, label=m
            ))
        handles.append(mlines.Line2D(
            [], [], color='k', marker='X', linestyle='None',
            markersize=6, label='Not significant'
        ))
        ax.legend(handles=handles, loc='upper right', fontsize=text_size)

        # Table data in reversed order to match plot top-down
        table_data = []
        row_labels = []
        # for param in reversed(params):
        for param in reversed(params):
            for mod in ['Unadjusted', 'Adjusted']:
                subset = self.df[
                    (self.df['study'] == param) & (self.df['Model'] == mod)
                    ]
                row_labels.append(f"{param} ({mod})")
                if not subset.empty:
                    r = subset.iloc[0]
                    table_data.append([
                        f"{r['OR']:.2f}",
                        f"{r['LCL']:.2f}-{r['UCL']:.2f}",
                        f"{r['pvalue']:.3f}"
                    ])
                else:
                    table_data.append(['', '', ''])

        tbl = tbl_ax.table(
            cellText=table_data,
            cellLoc='center',
            colLabels=['OR', self.ci, 'p-value'],
            rowLabels=row_labels,
            bbox=[0, 0, 1, 1]
        )
        tbl_ax.axis('off')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(text_size)

        # Save/show
        if path_save:
            fig.savefig(path_save, dpi=300)
        plt.show()


# === Logistic models & OR table builders ===
def _fit_logistic(df: pd.DataFrame, outcome: str, exposure: str,
                  covariates: Optional[List[str]] = None,
                  cat_vars: Optional[List[str]] = None,) -> smf.logit:
    """
    Fit a logistic regression model after cleaning data and verifying binary outcome.

    Returns a statsmodels LogitResults object.
    """
    cols = [outcome, exposure] + (covariates or [])
    clean_df = df.dropna(subset=cols)
    if not set(clean_df[outcome].unique()).issubset({0, 1}):
        raise ValueError(f"Outcome '{outcome}' must be binary.")
    terms = [f"C({exposure})"] + [f"C({c})" if (cat_vars and c in cat_vars) else c
                                     for c in (covariates or [])]
    formula = f"{outcome} ~ " + " + ".join(terms)
    return smf.logit(formula, data=clean_df).fit(cov_type='HC1', disp=False)

def _extract_model_stats(model) -> pd.DataFrame:
    """
    Extract ORs, standard errors, 95% CI, p-values, N, BIC, PseudoR2.
    """
    return pd.DataFrame({
        'OR': np.exp(model.params).round(2),
        'SE': model.bse.round(3),
        'CI Lower': np.exp(model.conf_int()[0]).round(3),
        'CI Upper': np.exp(model.conf_int()[1]).round(3),
        'p-value': model.pvalues.round(3),
        'N': int(model.nobs),
        'BIC': model.bic.round(3),
        'PseudoR2': (1 - model.llf / model.llnull).round(3)
    })

def build_or_table(df: pd.DataFrame,
                   outcome: str,
                   exposure: str,
                   adjust_vars: Optional[List[str]] = None,
                   cat_adjust: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Construct a combined odds-ratio table for a binary outcome and primary exposure,
    including both unadjusted and adjusted logistic regression models.

    Workflow:
    1. **Unadjusted model**: fits `outcome ~ exposure`.
    2. **Adjusted model**: fits `outcome ~ exposure + adjust_vars` (if provided).
    3. Extracts:
       - **OR** (odds ratio)
       - **SE** (standard error)
       - **CI Lower/Upper** (95% confidence limits)
       - **p-value** (two-sided)
       - **N** (effective sample size)
       - **BIC** (Bayesian Information Criterion)
       - **PseudoR2** (McFadden's pseudo-R²)
    4. Stacks the two model results into a single DataFrame with a `Model` column
       indicating `'Unadjusted'` or `'Adjusted'`.

    Args:
        df (pd.DataFrame): Input dataset containing outcome, exposure, and covariates.
        outcome (str): Name of binary outcome variable (must be coded 0/1).
        exposure (str): Name of the primary exposure variable.
        adjust_vars (List[str], optional): List of covariates for adjustment.
        cat_adjust (List[str], optional): Subset of `adjust_vars` to treat as categorical.

    Returns:
        pd.DataFrame: A table with one row per model parameter, columns:
            - Model: 'Unadjusted' or 'Adjusted'
            - OR, SE, CI Lower, CI Upper, p-value, N, BIC, PseudoR2
    """
    tbls = []
    for model_type, covs in [('Unadjusted', None), ('Adjusted', adjust_vars)]:
        m = _fit_logistic(df, outcome, exposure, covariates=covs, cat_vars=cat_adjust)
        stats = _extract_model_stats(m)
        stats['Model'] = model_type
        tbls.append(stats)
    res = pd.concat(tbls)
    res['Outcome'] = outcome
    cols = ['Model'] + [c for c in res.columns if c != 'Model']
    return res[cols]

def forest_plot(or_table: pd.DataFrame,
                alpha: float = 0.05,
                figsize: Tuple[int, int] = (14, 6),
                putput_path:pathlib.Path=None) -> None:
    """
    Produce a forest plot of ORs with robust CIs, color-coded by Model.
    Significant points (p<alpha and CI excludes 1) are marked.
    """
    labels = or_table.index.tolist()
    effects = or_table['OR'].tolist(); lcls = or_table['CI Lower'].tolist()
    ucls = or_table['CI Upper'].tolist(); pvals = or_table['p-value'].tolist()
    models = or_table['Model'].tolist()
    plotter = EffectMeasurePlot(label=labels,
                                effect_measure=effects,
                                lcl=lcls,
                                ucl=ucls,
                                p_value=pvals,
                                model=models,
                                alpha=alpha)
    plotter.set_labels(conf_int=f"{100*(1-alpha):.0f}% CI", scale='linear', center=1)
    plotter.set_colors(model_colors={'Unadjusted':'tab:blue','Adjusted':'tab:orange'},
                       linecolor='gray')
    plotter.plot(figsize=figsize, path_save=putput_path)

if __name__ == '__main__':
    # paths and data source
    from config.config import config, cleft_diag_codes, cleft_proc_codes, columns_categorical
    df =  pd.read_csv(config.get('data_pre_proc_files' ).get('pp_data_cases'), low_memory=True)
    output_path = config.get('results_path').get('results')

    # mapping categorical variables
    df['Race'] = df['race'].map({1:'White',2:'Asian',3:'Black',4:'Hispanic',5:'Other'})
    region_map = {1: 'North Central', 2: 'South Central', 3: 'East', 4: 'West', 5: 'Other'}
    df['hosp_region'] = df['hosp_region'].map(region_map)
    # define covariates, exposure and outcome
    adjust_vars=['age','sex']
    cat_adjust=['sex']
    exposure='Race'
    outcomes=[
        # 'complications',
              'prolonged_los']
    for outcome in outcomes:
        tbl=build_or_table(df,outcome,exposure,adjust_vars,cat_adjust)
        # display_dataframe_to_user(f'OR: {outcome} by Race',tbl)
        path_plot = output_path.joinpath(f'odds_rato_{outcome}.png')
        forest_plot(tbl, putput_path=path_plot, alpha=0.05, )

