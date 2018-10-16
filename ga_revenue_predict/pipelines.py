"""
Pipelines and transforms to be performed.
"""
import logging
import pandas as pd
from .Loader import Loader
from .Settings import USE_COLS, NUM_COLS, TARGET
from .CommonFunctions import store_df_and_dtypes, read_df_and_dtypes, deduplicate_repeated_sessionId
from sklearn_pandas import DataFrameMapper
from .sk_util import RemoveColumns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from .Paths import FULLPATH_DATA_TRAIN_PARSED, FULLPATH_DATA_TEST_PARSED
import datetime

def preprocessing():
    # Preprocessing steps, dropping unuse columns, fix dtypes, merge duplicated session IDs, impute missing values.
    from numpy import log1p
    # Load raw data
    loader = Loader()
    #df_train, df_test = loader.run(mode='raw', subsample=0.01)
    df_train, df_test = loader.run(mode='raw')

    # De-duplicate
    df_train = deduplicate_repeated_sessionId(df_train)
    df_test = deduplicate_repeated_sessionId(df_test)

    # remove unused columns
    rc = RemoveColumns(USE_COLS)
    df_train_rc = rc.transform(df_train)
    df_test_rc = rc.transform(df_test)

    train_data_mapper = DataFrameMapper([
        (NUM_COLS, SimpleImputer(strategy='constant', fill_value=0.0)),
        (TARGET, [SimpleImputer(strategy='constant', fill_value=0.0), FunctionTransformer(log1p)])
    ], input_df=True, df_out=True, default=None)

    test_data_mapper = DataFrameMapper([
        (NUM_COLS, SimpleImputer(strategy='constant', fill_value=0.0)),
    ], input_df=True, df_out=True, default=None)

    df_train_parsed = train_data_mapper.fit_transform(df_train_rc)
    df_test_parsed = test_data_mapper.fit_transform(df_test_rc)
    # print(train_data_mapper.transformed_names_)

    df_train_parsed.rename({'totals.bounces_totals.hits_totals.newVisits_totals.pageviews_0': 'totals.bounces',
                            'totals.bounces_totals.hits_totals.newVisits_totals.pageviews_1': 'totals.hits',
                            'totals.bounces_totals.hits_totals.newVisits_totals.pageviews_2': 'totals.newVisits',
                            'totals.bounces_totals.hits_totals.newVisits_totals.pageviews_3': 'totals.pageviews'},
                           axis=1, inplace=True)
    df_test_parsed.rename({'totals.bounces_totals.hits_totals.newVisits_totals.pageviews_0': 'totals.bounces',
                           'totals.bounces_totals.hits_totals.newVisits_totals.pageviews_1': 'totals.hits',
                           'totals.bounces_totals.hits_totals.newVisits_totals.pageviews_2': 'totals.newVisits',
                           'totals.bounces_totals.hits_totals.newVisits_totals.pageviews_3': 'totals.pageviews'},
                          axis=1, inplace=True)
    df_train_parsed = df_train_parsed.astype(df_train_rc.dtypes.to_dict())
    df_test_parsed = df_test_parsed.astype(df_test_rc.dtypes.to_dict())

    df_train_parsed = df_train_parsed.astype({'totals.bounces': int,
                                              'totals.hits': int,
                                              'totals.newVisits': int,
                                              'totals.pageviews': int,
                                              'totals.transactionRevenue': float})
    df_test_parsed = df_test_parsed.astype({'totals.bounces': int,
                                            'totals.hits': int,
                                            'totals.newVisits': int,
                                            'totals.pageviews': int})

    print(df_train_parsed.head(10))
    print(df_train_parsed.dtypes)

    store_df_and_dtypes(df_train_parsed, path=FULLPATH_DATA_TRAIN_PARSED, index=False)
    store_df_and_dtypes(df_test_parsed, path=FULLPATH_DATA_TEST_PARSED, index=False)
    return 0


def read_parsed_data():
    df_train = read_df_and_dtypes(path=FULLPATH_DATA_TRAIN_PARSED)
    df_test = read_df_and_dtypes(path=FULLPATH_DATA_TEST_PARSED)
    return df_train, df_test


def drop_bounce_clients(df_train, df_test):
    """
    Discard all clients without any non-bounce session.
    :param df_train:
    :param df_test:
    :return:
    """
    df = pd.concat([df_train, df_test], axis=0, sort=True)  # concat train and test set
    logging.info("Total number of unique clients: {}".format(df['fullVisitorId'].nunique()))
    df_lite = df[['fullVisitorId', 'sessionId', 'totals.bounces']]
    # count no. of sessions and no. of bounces
    df_grp = df_lite.groupby('fullVisitorId').agg({'sessionId': 'count', 'totals.bounces': 'sum'}).reset_index()
    lst_bounce_clients = df_grp[df_grp['sessionId'] == df_grp['totals.bounces']]['fullVisitorId'].tolist()
    logging.info("{} bounced users found, discard.".format(len(lst_bounce_clients)))
    df_no_bounce = df[~df['fullVisitorId'].isin(lst_bounce_clients)]
    df_bounce = df[df['fullVisitorId'].isin(lst_bounce_clients)]
    # Split back to train and test set
    df_train_out = df_no_bounce[df_no_bounce['date'].dt.date <= datetime.date(2017, 8, 1)]
    df_test_out = df_no_bounce[df_no_bounce['date'].dt.date > datetime.date(2017, 8, 1)]\
        .drop('totals.transactionRevenue', axis=1)
    return df_train_out, df_test_out, df_bounce


def split_by_returning(df_train, df_test):
    """
    Split out clients by total number of sessions (single or >=2)
    :param df_train:
    :param df_test:
    :return:
    """
    df = pd.concat([df_train, df_test], axis=0, sort=True)
    df_grp = df.groupby('fullVisitorId').agg({'sessionId': 'count'}).reset_index()
    lst_single = df_grp[df_grp['sessionId'] > 1]['fullVisitorId'].unique().tolist()
    df_single = df[df['fullVisitorId'].isin(lst_single)]
    df_return = df[~df['fullVisitorId'].isin(lst_single)]
    logging.info("Total # of clients with only 1 session: {}".format(df_single['fullVisitorId'].nunique()))
    logging.info("Total # of clients with more than 1 session: {}".format(df_return['fullVisitorId'].nunique()))
    df_single_train = df_single[df_single['date'].dt.date <= datetime.date(2017, 8, 1)]
    df_single_test = df_single[df_single['date'].dt.date > datetime.date(2017, 8, 1)]\
        .drop('totals.transactionRevenue', axis=1)
    df_return_train = df_return[df_return['date'].dt.date <= datetime.date(2017, 8, 1)]
    df_return_test = df_return[df_return['date'].dt.date > datetime.date(2017, 8, 1)]\
        .drop('totals.transactionRevenue', axis=1)
    logging.info("Split of train/test 1 session clients: {}::{}"
                 .format(df_single_train['fullVisitorId'].nunique(), df_single_test['fullVisitorId'].nunique()))
    logging.info("Split of train/test returning clients: {}::{}"
                 .format(df_return_train['fullVisitorId'].nunique(), df_return_test['fullVisitorId'].nunique()))
    return df_single_train, df_single_test, df_return_train, df_return_test
