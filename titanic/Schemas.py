"""
Module which stores the default schemas of tables.
"""

SCHEMAS = {
    'train': {
        'PassengerId': str,
        'Survived': int,
        'Pclass': str,
        'Name': str,
        'Sex': str,
        'Age': float,
        'SibSp': float,
        'Parch': float,
        'Ticket': str,
        'Fare': float,
        'Cabin': str,
        'Embarked': str
    },
    'test': {
        'PassengerId': str,
        'Pclass': str,
        'Name': str,
        'Sex': str,
        'Age': float,
        'SibSp': float,
        'Parch': float,
        'Ticket': str,
        'Fare': float,
        'Cabin': str,
        'Embarked': str
    },
    'submission': {
        'PassengerId': str,
        'Survived': int
    }
}