"""

"""
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ..CommonFunctions import deduplicate_repeated_sessionId, drop_bounced_sessions
from ..sk_util import ModifiedLabelEncoder, plot_confusion_matrix
import sklearn
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn_pandas import DataFrameMapper, gen_features
from sklearn.model_selection import train_test_split
import autosklearn.classification
from pprint import pprint
import os
from ..Paths import PATH_DIR_DATA, PATH_DIR_MODEL


df_train = pd.read_csv(os.path.join(PATH_DIR_DATA, 'train_filled.csv'), low_memory=False)
df_test = pd.read_csv(os.path.join(PATH_DIR_DATA, 'test_filled.csv'), low_memory=False)

# Fix duplicated sessionsId
df_train_dedupped = deduplicate_repeated_sessionId(df_train)
df_test_dedupped = deduplicate_repeated_sessionId(df_test)

# By common sense, bounced sessions do not have purchase. No need to explicitly predict them.
df_train_flt = drop_bounced_sessions(df_train_dedupped)
df_test_flt = drop_bounced_sessions(df_test_dedupped)

# Do a easy model first, predict session level purchase
# Cassify purchase or not, so create a new target
df_train_flt['isPurchase'] = df_train_flt['totals.transactionRevenue'] > 0.0
df_train_flt['isPurchase'] = df_train_flt['isPurchase'].map({True: 1, False:0})
df_train_flt.drop('totals.transactionRevenue', axis=1, inplace=True)

TARGET = df_train_flt['isPurchase'].copy()
TRAIN = df_train_flt.drop('isPurchase', axis=1).copy()

# 1. Drop some columns
use_cols = ['channelGrouping', 'visitNumber',
            'device.deviceCategory', 'device.isMobile',
            'totals.hits', 'totals.newVisits', 'totals.pageviews',
            'trafficSource.isTrueDirect'
           ]
TRAIN = TRAIN[use_cols]

# Preprocessing
# Pipeline
feature_cat = gen_features(columns=['channelGrouping', 'device.deviceCategory'],
                           classes=[ModifiedLabelEncoder,
                                    OneHotEncoder
                                    ]
                           )
feature_num = gen_features(columns=[['visitNumber'],
                                    ['device.isMobile'],
                                    ['totals.hits'], ['totals.newVisits'], ['totals.pageviews'],
                                    ['trafficSource.isTrueDirect']],
                           classes=[StandardScaler])
mapper = DataFrameMapper(
    feature_cat + feature_num,
    input_df=True, df_out=True)
TRAIN_preprocessed = mapper.fit_transform(TRAIN.copy())

# Spilt train/test sets
X_train, X_test, y_train, y_test = train_test_split(TRAIN_preprocessed, TARGET, random_state=26)

# Auto-sklearn
automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=60*60*48,
        per_run_time_limit=60*60,
        #tmp_folder=os.path.join('tmp', 'autosklearn_cv_example_tmp'),
        #output_folder=os.path.join('tmp', 'autosklearn_cv_example_out'),
        #delete_tmp_folder_after_terminate=True,
        resampling_strategy='holdout-iterative-fit',
        #resampling_strategy_arguments={'folds': 10},
        #ml_memory_limit=2048
    )
# Train to find best parameters
automl.fit(X_train, y_train, dataset_name='garp', metric=autosklearn.metrics.roc_auc)
automl.refit(X_train.copy(), y_train.copy())

# Save refit model
pickle.dump(automl, os.path.join(PATH_DIR_MODEL, 'automl_1'))

print(automl.show_models())
#pprint(automl.cv_results_)
print(automl.sprint_statistics())

predictions = automl.predict(X_test)
print("accuracy score", sklearn.metrics.accuracy_score(y_test, predictions))
print("F1 score", sklearn.metrics.f1_score(y_test, predictions))
print("precision score", sklearn.metrics.precision_score(y_test, predictions))
print("recall score", sklearn.metrics.recall_score(y_test, predictions))
print("ROC score", sklearn.metrics.roc_auc(y_test, predictions))

# Plot confusion matrix
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, predictions)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
class_names = y_test.unique()
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()
