import pandas as pd


def validate_df(df: pd.DataFrame, df_cols: list):
    cols = df.columns
    missing_cols = []
    for col in df_cols:
        if col not in cols:
            missing_cols.append(col)
    assert len(missing_cols) == 0, f'The following columns are missing in the ingest_df: {missing_cols}'

    for col in df_cols:
        na_count = df[col].isna().sum()
        assert na_count == 0, f'The column {col} has {na_count} NA values'


def validate_column(df: pd.DataFrame, column_name: str, approved_values: list):
    vals = df[column_name].unique()
    bad_vals = []
    for val in vals:
        if val not in approved_values:
            bad_vals.append(val)
    assert len(bad_vals) == 0, f'The values are not approved: {bad_vals}, the approved list is {approved_values}'