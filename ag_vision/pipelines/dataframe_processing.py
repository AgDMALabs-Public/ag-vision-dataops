import pandas as pd
import os


def keep_records_with_missing_metadata_files(
    df: pd.DataFrame,
    path_col: str = "metadata_path"
    )-> pd.DataFrame:

    """
    Adds a boolean column 'metatada_file_exists' to df, then returns
    only the rows where the file does NOT already exist.
    """
    df = df.copy()
    df["metatada_file_exists"] = df[path_col].apply(os.path.exists)
    return df[df["metatada_file_exists"] == False]