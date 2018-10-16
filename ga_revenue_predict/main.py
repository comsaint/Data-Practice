import logging

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    from .pipelines import preprocessing, read_parsed_data, drop_bounce_clients, split_by_returning
    #preprocessing()
    df_train, df_test = read_parsed_data()
    df_train, df_test, df_bounce = drop_bounce_clients(df_train, df_test)
    df_single_train, df_single_test, df_return_train, df_return_test = split_by_returning(df_train, df_test)

    df_single_train.to_csv('train_single_session.csv', index=False)
    df_single_test.to_csv('test_single_session.csv', index=False)
    #df_bounce.to_csv('bounce.csv', index=False)
    #print(df_train.head())
    #print(df_train.dtypes)
