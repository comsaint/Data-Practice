"""
This script is a simple LR model with minimal feature engineering.
"""

import logging
from ..Loader import Loader
from ..Paths import DICT_PATHS
import pandas as pd
from sklearn.pipeline import make_pipeline
from ..sk_util import CategoricalEncoder
from sklearn.preprocessing import Imputer, FunctionTransformer, StandardScaler
from sklearn.linear_model import LogisticRegression

loader = Loader()
df_train = loader.read_original_data(table_code='train')
df_test = loader.read_original_data(table_code='test')

# Consider only a subset of columns
df_train.set_index('PassengerId', inplace=True)
df_test.set_index('PassengerId', inplace=True)

USE_COLS = ['Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
TARGET = ['Survived']
X_train = df_train[USE_COLS].copy()
y_train = df_train[TARGET].copy()
X_test = df_test[USE_COLS].copy()

# Define categories
col_categories = ['Sex', 'Embarked']
categories = dict()
for cat_col in col_categories:
    # Ignore nan (we will impute them before encoding)
    cats = X_train[~df_train[cat_col].isnull()][cat_col].unique().tolist()
    categories.update({cat_col: cats})
print(list(categories.keys()))

# Columns that needs standard scaling
col_std_scl = ['Age', 'SibSp', 'Parch', 'Fare']

# Pipelines
# Fit and transform all categories
X_train_cat = X_train[list(categories.keys())]
pipe_cat = make_pipeline(
        CategoricalEncoder(categories),
        FunctionTransformer(pd.get_dummies, validate=False),
)
X_train_cat_fit = pipe_cat.fit_transform(X_train_cat)

# Fit and transform all scalars
X_train_num = X_train[col_std_scl]
pipe_num = make_pipeline(
    Imputer(strategy='mean'),
    StandardScaler()
)
X_train_num_fit = pipe_num.fit_transform(X_train_num)
X_train_num_fit = pd.DataFrame(X_train_num_fit, columns=col_std_scl, index=X_train_num.index)

# Merge the DataFrame back
X_train_fit = X_train_cat_fit.join(X_train_num_fit)

# Training
pipe_train = make_pipeline(
    LogisticRegression()
)
pipe_train.fit(X_train_fit, y_train)
print(pipe_train.score(X_train_fit, y_train))

#print(X_train_fit.describe(include='all'))
#print(df_train.head())
#print(df_test.head())


