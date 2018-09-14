"""
Module which stores paths of directory/files.
"""
import os

''' 1. Original data'''
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

PATH_DIR_DATA = os.path.join(ROOT_DIR, r'./data')
# Raw data set
FILE_DATA_TRAIN = r'train.csv'
FILE_DATA_TEST = r'test.csv'
FILE_DATA_SAMPLE_SUBMISSION = r'sample_submission.csv'

FULLPATH_DATA_TRAIN = os.path.join(PATH_DIR_DATA, FILE_DATA_TRAIN)
FULLPATH_DATA_TEST = os.path.join(PATH_DIR_DATA, FILE_DATA_TEST)
FULLPATH_DATA_SAMPLE_SUBMISSION = os.path.join(PATH_DIR_DATA, FILE_DATA_SAMPLE_SUBMISSION)

# Parsed data set
FILE_DATA_TRAIN_PARSED = r'train_parsed.csv'
FILE_DATA_TEST_PARSED = r'test_parsed.csv'
FULLPATH_DATA_TRAIN_PARSED = os.path.join(PATH_DIR_DATA, FILE_DATA_TRAIN_PARSED)
FULLPATH_DATA_TEST_PARSED = os.path.join(PATH_DIR_DATA, FILE_DATA_TEST_PARSED)

''' 2. Output (prediction) '''
PATH_DIR_PREDICTIONS = os.path.join(ROOT_DIR, r'predictions')
FILE_PREDICTIONS = r'predictions.csv'
FULLPATH_PREDICTIONS = os.path.join(PATH_DIR_PREDICTIONS, FILE_PREDICTIONS)

DICT_PATHS = {
    'train': FULLPATH_DATA_TRAIN,
    'test': FULLPATH_DATA_TEST,
    'train_parsed': FULLPATH_DATA_TRAIN_PARSED,
    'test_parsed': FULLPATH_DATA_TEST_PARSED,
    'sample_submission': FULLPATH_DATA_SAMPLE_SUBMISSION,
    'predictions': FULLPATH_PREDICTIONS
}
