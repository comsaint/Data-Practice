"""
Rewrite the pipeline of simple_model_3, and test more models. Also keep record on Comet.ml
"""
from sklearn_pandas import DataFrameMapper, CategoricalImputer
from sklearn_pandas import gen_features

import logging
from ..Loader import Loader
from ..Paths import DICT_PATHS
import pandas as pd

from ..sk_util import ModifiedLabelEncoder
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.preprocessing import OneHotEncoder

loader = Loader()
df_train = loader.read_original_data(table_code='train')
df_test = loader.read_original_data(table_code='test')

# Consider only a subset of columns
df_train.set_index('PassengerId', inplace=True)
df_test.set_index('PassengerId', inplace=True)

USE_COLS = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
TARGET = ['Survived']
X_train = df_train[USE_COLS].copy()
y_train = df_train[TARGET].copy().values.reshape(-1,)
X_test = df_test[USE_COLS].copy()

# Preprocessing
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

# Training
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from sklearn.model_selection import RandomizedSearchCV
import scipy, itertools

# Hyperparameters
dict_param_grid = {
    'KNeighborsClassifier': {
        'n_neighbors': scipy.stats.randint(2, 20),
        'weights': ['uniform', 'distance'],
        'n_jobs': [-1]
    },
    'GaussianNB': None,
    'MLPClassifier': {
        'hidden_layer_sizes': [x for x in itertools.product((10, 20, 30, 50, 100), repeat=3)],
        'activation': ['relu', 'tanh'],
        'alpha': scipy.stats.expon(scale=100),
        'max_iter': [200, 1000],
        'early_stopping': [True, False]
    },
    'GradientBoostingClassifier': {'learning_rate': scipy.stats.uniform(loc=0.01, scale=0.15),
                                    'max_depth': scipy.stats.randint(low=2, high=X_train_fit.shape[1]),
                                    'min_samples_split': scipy.stats.uniform(loc=0.001, scale=0.02),
                                    'min_samples_leaf': scipy.stats.uniform(loc=0.01, scale=0.2),
                                    'subsample': scipy.stats.uniform(loc=0.5, scale=0.4),
                                    'max_features': ['auto', 'sqrt', 'log2', None]},
    'LogisticRegression': {'penalty': ['l1', 'l2'],
                           'C': scipy.stats.expon(scale=10)},
    'RandomForestClassifier': {'n_estimators': scipy.stats.randint(5, 1000),
                                'max_features': ['auto', 'sqrt'],
                                'max_depth': scipy.stats.randint(10, 100),
                                'min_samples_leaf': scipy.stats.randint(1, 4),
                                'min_samples_split': scipy.stats.randint(2, 10),
                                'bootstrap': [True, False]},
    'SVC': {'kernel': ['rbf'],
            'C': scipy.stats.expon(scale=100),
            'gamma': scipy.stats.expon(scale=100),
            'class_weight': ['balanced'],
            'probability': [True]}
}

models = [
    MLPClassifier(),
    KNeighborsClassifier(),
    GaussianNB(),
    GradientBoostingClassifier(),
    LogisticRegression(),
    RandomForestClassifier(),
    SVC(),
]

lst_best_models = []
for model in models:
    model_name, param = model.__class__.__name__, dict_param_grid[model.__class__.__name__]
    if param is not None:
        clf = RandomizedSearchCV(model, param_distributions=param, cv=5, verbose=0, n_jobs=-1, n_iter=200)
        print("Training model: {}".format(model.__class__.__name__))
        lst_best_models.append((model_name, clf.fit(X_train_fit, y_train)))
    else:
        lst_best_models.append((model_name, model.fit(X_train_fit, y_train)))

# Ensemble of models
from sklearn.ensemble import VotingClassifier
from sklearn.cross_validation import cross_val_score
eclf = VotingClassifier(estimators=lst_best_models, voting='soft')
eclf.fit(X_train_fit, y_train)
scores = cross_val_score(eclf, X_train_fit, y_train, cv=5, scoring='accuracy')
print(scores)
'''
clf = RandomizedSearchCV(model, param_distributions=params, cv=5, verbose=1, n_jobs=-1, n_iter=100)

logging.info("Training...")
best_model = clf.fit(X_train_fit, y_train)

# Print best results (rank by high test score and low std)
train_result = pd.DataFrame(clf.cv_results_)
train_result.sort_values(by=['mean_test_score', 'std_test_score'], ascending=[False, True], inplace=True)
print(train_result[['mean_test_score', 'std_test_score']].head())

for param in params:
    print('Parameter: {}, best value={}'.format(param, best_model.best_estimator_.get_params()[param]))
#print("Accuracy: {}".format(best_model.cv_results_['mean_test_score']))


pipe_model = make_pipeline(
    RandomizedSearchCV(LogisticRegression(),
                       param_distributions=param_grid_LR,
                       cv=5)
)



pipe_model.fit(X_train_fit, y_train)
print(pipe_model.score(X_train_fit, y_train))
'''

# Apply on test set
X_test_txf = mapper.transform(X_test.copy())
logging.info("Predicting...")
y_test_predict = pd.DataFrame(eclf.predict(X_test_txf), index=X_test.index, columns=TARGET)

# Write prediction
y_test_predict.to_csv(DICT_PATHS['predictions'], encoding='utf-8', index=True)

'''
# Feature selection
from sklearn.feature_selection import SelectKBest, mutual_info_classif
kselector = SelectKBest(mutual_info_classif, k=3)
X_new = kselector.fit_transform(X_train_fit, y_train)

print(pd.DataFrame([kselector.scores_], columns=X_train_fit.columns))

mapper_fs = DataFrameMapper([
    (list(X_train_fit.columns), SelectKBest(mutual_info_classif, k=2))
    ], input_df=True, df_out=True
)
fs = mapper_fs.fit_transform(X_train_fit, y_train)

#print(fs.head())
'''
