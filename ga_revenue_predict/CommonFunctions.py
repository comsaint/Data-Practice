import pandas as pd


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
