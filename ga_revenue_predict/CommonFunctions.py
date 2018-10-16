import pandas as pd
import json


def deduplicate_repeated_sessionId(df):
    import numpy as np
    dedup_method = dict()
    for col in df.columns:
        if col in ['totals.bounces']:
            dedup_method.update({col: np.min})
        elif col in ['totals.hits', 'totals.pageviews', 'totals.transactionRevenue']:
            dedup_method.update({col: np.sum})
        elif col in ['totals.newVisits']:
            dedup_method.update({col: np.max})
        else:
            dedup_method.update({col: np.min})
    df_dup_id = df[['sessionId', 'fullVisitorId']].groupby('sessionId').size().reset_index()
    df_dup_id.columns = ['sessionId', 'cnt']
    list_dup_id = df_dup_id[df_dup_id['cnt'] > 1]['sessionId'].tolist()

    # Split
    df_dup_sessions = df[df['sessionId'].isin(list_dup_id)]
    df_non_dup_sessions = df[~df['sessionId'].isin(list_dup_id)]

    df_deduped = df_dup_sessions.groupby('sessionId').agg(dedup_method).drop('sessionId', axis=1).reset_index()
    return pd.concat([df_non_dup_sessions, df_deduped], sort=False)


def drop_bounced_sessions(df):
    return df[df['totals.bounces'] == 0].copy()


def make_dtypes_dict(df):
    """
    Given a Pandas DataFrame, create a dict of columns: dtypes.
    :param df:
    :return:
    """
    res = df.dtypes.to_frame('dtypes').reset_index()  # create a DF of dtypes
    return res.set_index('index')['dtypes'].astype(str).to_dict()  # convert to DICT


def store_dtypes_json(df, path):
    """
    Given a Pandas DataFrame, create a dict of columns: dtypes, and save to `path`.
    :param df:
    :param path:
    :return:
    """
    d = make_dtypes_dict(df)
    with open(path, 'w') as f:
        json.dump(d, f)
    return None


def read_dtypes_json(path):
    """
    Read a JSON file from `path`.
    :param path:
    :return:
    """
    with open(path, 'r') as f:
        return json.load(f)


def store_df_and_dtypes(df, path, **kwargs):
    """

    :param df:
    :param path:
    :return:
    """
    df.to_csv(path, **kwargs)
    path_dtypes = path.replace(r'.csv', r'.json')
    store_dtypes_json(df, path_dtypes)
    return None


def read_df_and_dtypes(path, **kwargs):
    """

    :param path:
    :param kwargs:
    :return:
    """
    path_dtypes = path.replace(r'.csv', r'.json')
    kwargs.update({'dtype': read_dtypes_json(path_dtypes)})
    col_dt = []
    for col_name, col_dtype in kwargs['dtype'].items():
        if 'datetime64' in col_dtype:
            col_dt.append(col_name)
    if col_dt != []:
        for col in col_dt:
            del kwargs['dtype'][col]
        kwargs.update({'parse_dates': col_dt})
    return pd.read_csv(path, **kwargs)
