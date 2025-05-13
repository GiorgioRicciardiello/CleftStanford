"""
Table 1 generator

This script builds a stratified “Table 1” that summarizes:
  • Categorical variables:
      – Counts and column-percentages via pandas.crosstab
      – Pearson’s χ² test of independence (scipy.stats.chi2_contingency)
  • Continuous/Ordinal variables:
      – Descriptive statistics (mean, standard deviation, median, Q1, Q3) via pandas.groupby + agg
      – Kruskal–Wallis H-test for group differences (scipy.stats.kruskal)

Functions:
  – _summarize_categorical: crosstab + χ²
  – _summarize_continuous: group stats + Kruskal–Wallis
  – build_table1: orchestrates filtering, calls summaries, and concatenates results
"""
import pandas as pd
from config.config import config, cleft_diag_codes, cleft_proc_codes, columns_categorical
import numpy as np
from typing import Dict, Callable, List, Optional
from tabulate import tabulate
from scipy.stats import chi2_contingency, kruskal
import statsmodels.formula.api as smf


def _summarize_categorical(
    df: pd.DataFrame,
    column: str,
    group_col: str = 'race'
) -> pd.DataFrame:
    """
    Summarize a categorical variable by counts and percentages across groups, with chi-square p-value.

    Args:
        df: Input DataFrame containing data.
        column: Name of the categorical column to summarize.
        group_col: Name of the grouping column (e.g., 'race').

    Returns:
        A DataFrame with levels of `column` as rows, group counts and percentages as cells, and p-value.
    """
    # Crosstab counts
    counts = pd.crosstab(df[column], df[group_col], dropna=False)
    # Percentages
    percents = counts.div(counts.sum(axis=0), axis=1).mul(100).round(1)
    # Combine counts and percentages
    summary = counts.astype(str) + ' (' + percents.astype(str) + '%)'

    # Compute chi-square p-value
    chi2, pval, _, _ = chi2_contingency(counts.fillna(0))
    summary['P-Value'] = ''
    summary.iloc[0, -1] = pval

    # Rename index for clarity
    summary.index = [f"{column}: {level}" for level in summary.index]
    return summary


def _summarize_continuous(
    df: pd.DataFrame,
    column: str,
    aggfunc: Callable = np.median,
    group_col: str = 'race'
) -> pd.DataFrame:
    """
    Summarize a continuous or ordinal variable by a specified statistic across groups, with Kruskal-Wallis p-value.

    Args:
        df: Input DataFrame containing data.
        column: Name of the continuous/ordinal column to summarize.
        aggfunc: Aggregation function to apply (e.g., np.median, np.mean).
        group_col: Name of the grouping column (e.g., 'race').

    Returns:
        A DataFrame with the statistic per group and p-value.
    """
    # compute per-group metrics
    stats = df.groupby(group_col)[column].agg([
        'mean',
        'std',
        'median',
        lambda x: x.quantile(0.25),
        lambda x: x.quantile(0.75)
    ])
    stats.columns = ['mean', 'std', 'median', 'q1', 'q3']

    # build formatted summary strings for each group
    summary_dict = {
        grp: f"{row['mean']:.1f} ± {row['std']:.1f}; "
             f"{row['median']:.1f} [{row['q1']:.1f}–{row['q3']:.1f}]"
        for grp, row in stats.iterrows()
    }
    # kruskal-wallis p-value
    groups = [g[column].dropna() for _, g in df.groupby(group_col)]
    pval = kruskal(*groups)[1]

    # assemble into a single-row DataFrame
    summary_df = pd.DataFrame(summary_dict, index=[column])
    summary_df['P-Value'] = pval
    return summary_df


def build_table1(
    df: pd.DataFrame,
    strata: str,
    categorical: Dict[str, str],
    continuous: Dict[str, str],
    ordinal: Dict[str, str],
    filter_col: str = None
) -> pd.DataFrame:
    """
    Build Table 1 summarizing categorical, continuous, and ordinal variables by race.

    Args:
        df: Raw DataFrame.
        race_map: Mapping from numeric race codes to labels.
        categorical: Dict of label to column name for categorical variables.
        continuous: Dict of label to column name for continuous variables.
        ordinal: Dict of label to column name for ordinal variables.
        filter_col: Optional column name to filter df (e.g., 'included').

    Returns:
        A concatenated DataFrame representing Table 1.
    """
    # Optional filtering
    if filter_col:
        df = df[df[filter_col] == 1]

    parts = []
    # 1) Number of patients
    n = df.groupby(strata).size().to_frame().T
    n.index = ['Number of patients']
    parts.append(n)

    # 2) Categorical summaries
    for label, col in categorical.items():
        part = _summarize_categorical(df, col)
        parts.append(part)

    # 3) Continuous summaries (mean ± SD; median [Q1–Q3])
    for label, col in continuous.items():
        part = _summarize_continuous(df, col)
        part.index = [label]  # use descriptive label
        parts.append(part)

    # 4) Ordinal summaries (same as continuous)
    for label, col in ordinal.items():
        part = _summarize_continuous(df, col)
        part.index = [label]
        parts.append(part)

    # 5) concatenate everything
    table1 = pd.concat(parts)
    return table1