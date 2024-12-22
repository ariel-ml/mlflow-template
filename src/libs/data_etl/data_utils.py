from typing import List, Tuple
import pandas as pd
import numpy as np

TARGET_COLUMN = "target"


def concat(X: pd.DataFrame | np.ndarray, y: pd.Series, columns: str) -> pd.DataFrame:
    """
    Concatenates the given DataFrame or numpy array with the given Series.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        The DataFrame or numpy array to be concatenated.
    y : pd.Series
        The Series to be concatenated.
    columns : str
        The columns of the resulting DataFrame.

    Returns
    -------
    pd.DataFrame
        The concatenated DataFrame.
    """
    return pd.DataFrame(np.c_[X, y], columns=columns)


def save_data(data: pd.DataFrame, data_path: str):
    """
    Saves the given DataFrame to a parquet file at the given path.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame to be saved.
    data_path : str
        The path to the parquet file to be saved.
    """
    data.to_parquet(data_path)


def load_data(
    data_path: str, target: str = None
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Loads data from a parquet file and returns it as a tuple of (X, y, columns).

    Parameters
    ----------
    data_path : str
        The path to the parquet file.
    target : str, optional
        The name of the target variable. If not provided, the last column is used.

    Returns
    -------
    X : pd.DataFrame
        The data without the target variable.
    y : pd.Series
        The target variable.
    columns : List[str]
        The names of the columns in the data.
    """
    data = pd.read_parquet(data_path)
    if target:
        X = data.drop(columns=target, axis=1)
        y = data[target]
    else:
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
    return X, y, data.columns.tolist()
