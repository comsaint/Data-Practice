import logging

if __name__ == '__main__':
    from .pipelines import preprocessing, read_parsed_data
    #preprocessing()
    df_train, df_test = read_parsed_data()
    print(df_train.head())
    print(df_train.dtypes)
