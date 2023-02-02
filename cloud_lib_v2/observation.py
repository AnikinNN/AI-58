import warnings
from pathlib import Path

import pandas as pd


def init_observation(table_path: Path):
    """
        reads table on table_path
        creates new DataFrame with observations
        fields columns:
            observation_datetime
        sorts by observation_datetime
    """
    df_observation = read_table(table_path)
    df_observation = rename_datetime_column(df_observation)
    df_observation = drop_empty_rows(df_observation)
    df_observation = to_datetime(df_observation)
    df_observation = rename_cloud_type_column(df_observation)
    df_observation = rename_tcc_column(df_observation)

    df_observation.sort_values(by="observation_datetime", inplace=True)
    return df_observation


def read_table(table_path: Path) -> pd.DataFrame:
    if table_path.name.endswith("xlsx"):
        df_observation = pd.read_excel(table_path, )
    elif table_path.name.endswith("csv"):
        df_observation = pd.read_csv(table_path)
    else:
        raise ValueError(f'{table_path=}. Extension must be xlsx or csv')

    return df_observation


def rename_datetime_column(df_observation: pd.DataFrame) -> pd.DataFrame:
    possible_datetime = (
        'Date_Time (UTC)',
        'Date_Time',
        'dt',
    )
    return rename_column(df_observation, 'observation_datetime', possible_datetime)


def rename_cloud_type_column(df_observation: pd.DataFrame) -> pd.DataFrame:
    possible_names = (
        'CTypes',
        'Types of clouds',
    )
    return rename_column(df_observation, 'cloud_type', possible_names)


def rename_tcc_column(df_observation: pd.DataFrame) -> pd.DataFrame:
    possible_names = (
        'TCC',
        'TCC (общее количество облаков)',
    )
    return rename_column(df_observation, 'TCC', possible_names)


def rename_column(df_observation: pd.DataFrame, new_name: str, possible_names: tuple[str, ...]) -> pd.DataFrame:
    old_name = [i for i in df_observation.columns if i in possible_names]

    if len(old_name) != 1:
        raise RuntimeError(f'There are not a single {old_name=}')
    else:
        old_name = old_name[0]

    df_observation = df_observation.rename(columns={old_name: new_name})
    return df_observation


def drop_empty_rows(df_observation: pd.DataFrame) -> pd.DataFrame:
    df_observation = df_observation.dropna(axis='index', how='all')
    df_observation = df_observation[df_observation.observation_datetime != '<EOF>']
    return df_observation


def to_datetime(df_observation: pd.DataFrame) -> pd.DataFrame:
    dt = pd.to_datetime(df_observation.observation_datetime, format='%Y-%m-%d %H:%M:%S', errors='coerce')
    is_na = dt.isna()

    if is_na.sum() > 0:
        warnings.warn(f'There are {is_na.sum()} mistakes in dates. They are dropped\n'
                      f'{df_observation[is_na].observation_datetime}')

    df_observation['observation_datetime'] = dt
    df_observation = df_observation[~is_na]

    return df_observation
