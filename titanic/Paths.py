"""
Module which stores paths of directory/files.
"""
import os

''' 1. Original data'''
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

PATH_DIR_DATA = os.path.join(ROOT_DIR, r'./data')
FILE_DATA_TRAIN = r'train.csv'
FILE_DATA_TEST = r'test.csv'
FILE_DATA_GENDER_SUBMISSION = r'gender_submission.csv'

FULLPATH_DATA_TRAIN = os.path.join(PATH_DIR_DATA, FILE_DATA_TRAIN)
FULLPATH_DATA_TEST = os.path.join(PATH_DIR_DATA, FILE_DATA_TEST)
FULLPATH_DATA_GENDER_SUBMISSION = os.path.join(PATH_DIR_DATA, FILE_DATA_GENDER_SUBMISSION)

''' 2. Output (prediction) '''
PATH_DIR_PREDICTIONS = os.path.join(ROOT_DIR, r'predictions')
FILE_PREDICTIONS = r'predictions.csv'
FULLPATH_PREDICTIONS = os.path.join(PATH_DIR_PREDICTIONS, FILE_PREDICTIONS)

DICT_PATHS = {
    'train': FULLPATH_DATA_TRAIN,
    'test': FULLPATH_DATA_TEST,
    'gender_submission': FULLPATH_DATA_GENDER_SUBMISSION,
    'predictions': FULLPATH_PREDICTIONS
}
