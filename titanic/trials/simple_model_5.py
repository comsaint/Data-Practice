"""
Same pipeline as simple_model_4, but use auto-sklearn for model training.
"""
from sklearn_pandas import DataFrameMapper, CategoricalImputer
from sklearn_pandas import gen_features

import os
import logging
from ..Loader import Loader
from ..Paths import DICT_PATHS, PATH_DIR_DATA
import pandas as pd

from ..sk_util import ModifiedLabelEncoder

loader = Loader()
df_train = loader.read_original_data(table_code='train')
df_test = loader.read_original_data(table_code='test')

# Consider only a subset of columns
df_train.set_index('PassengerId', inplace=True)
df_test.set_index('PassengerId', inplace=True)

# Set data type
categorical_indicator = ['Pclass', 'Sex', 'Embarked']
df_train[categorical_indicator] = df_train[categorical_indicator].astype('category')
df_test[categorical_indicator] = df_test[categorical_indicator].astype('category')

USE_COLS = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
TARGET = ['Survived']
X_train = df_train[USE_COLS].copy()
y_train = df_train[TARGET].copy().values.reshape(-1,)
X_test = df_test[USE_COLS].copy()

# Preprocessing
from sklearn.preprocessing import OneHotEncoder, Imputer, StandardScaler
feature_cat = gen_features(columns=['Pclass', 'Sex', 'Embarked'],
                           classes=[CategoricalImputer,
                                    ModifiedLabelEncoder,
                                    OneHotEncoder
                                    ]
                           )
feature_num = gen_features(columns=[['Age'], ['SibSp'], ['Parch'], ['Fare']],
                           classes=[Imputer, StandardScaler])
mapper = DataFrameMapper(
    feature_cat + feature_num,
    input_df=True, df_out=True)

X_train_fit = mapper.fit_transform(X_train.copy())

# Make a copy
X_train_copy = X_train_fit.copy()
y_train_copy = y_train.copy()

# Training - use auto-sklearn
import sklearn.model_selection
import autosklearn.classification
automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=8*60*60,
        per_run_time_limit=300,
        tmp_folder=os.path.join(PATH_DIR_DATA, 'tmp', 'autosklearn_cv_example_tmp'),
        output_folder=os.path.join(PATH_DIR_DATA, 'tmp', 'autosklearn_cv_example_out'),
        delete_tmp_folder_after_terminate=True,
        resampling_strategy='cv',
        resampling_strategy_arguments={'folds': 5},
    )
# Train to find best parameters
automl.fit(X_train_fit, y_train, dataset_name='titanic')

# Train again with best models/parameters on the whole dataset again
automl.refit(X_train_copy, y_train_copy)

#print(automl.show_models())
#print(automl.cv_results_)
print(automl.sprint_statistics())
print("Accuracy score: ", sklearn.metrics.accuracy_score(y_train_copy, automl.predict(X_train_copy)))

# Apply on test set
logging.info("Predicting...")
X_test_txf = mapper.transform(X_test.copy())
y_test_predict = automl.predict(X_test_txf)
df_test_predict = pd.DataFrame(y_test_predict, index=X_test_txf.index, columns=TARGET)

# Write prediction
df_test_predict.to_csv(DICT_PATHS['predictions'], encoding='utf-8', index=True)
