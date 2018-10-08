"""
Module which stores the default schemas of tables.
"""


SCHEMAS = {
    'train': {
        'channelGrouping': str,
        'fullVisitorId': str,
        'sessionId': str,
        'socialEngagementType': str,
        'visitId': str,
        'visitNumber': int,
        'visitStartTime': int,
        'totals.bounces': float,
        'totals.hits': float,
        'totals.newVisits': float,
        'totals.pageviews': float,
        'totals.transactionRevenue': float
    },
    'train_parsed': {
        'device.isMobile': bool,
        'totals.bounces': int,
        'totals.hits': int,
        'totals.newVisits': int,
        'totals.pageviews': int,
        'totals.transactionRevenue': float,
        'channelGrouping': str,
        'fullVisitorId': str,
        'sessionId': str,
        'visitNumber': int,
        'visitStartTime': int,
        'trafficSource.adwordsClickInfo.page': str
    },
    'test': {
        'device.isMobile': bool,
        'totals.bounces': int,
        'totals.hits': int,
        'totals.newVisits': bool,
        'totals.pageviews': int,
        'totals.visits': int,
        'channelGrouping': str,
        'fullVisitorId': str,
        'sessionId': str,
        'visitNumber': int,
        'visitStartTime': int,
        'trafficSource.adwordsClickInfo.page': str
    },
    'test_parsed': {
        'device.isMobile': bool,
        'totals.bounces': float,
        'totals.hits': float,
        'totals.newVisits': float,
        'totals.pageviews': float,
        'totals.visits': float,
        'channelGrouping': str,
        'fullVisitorId': str,
        'sessionId': str,
        'socialEngagementType': str,
        'visitId': str,
        'visitNumber': float,
        'visitStartTime': float
    },
    'submission': {
        'fullVisitorId': str,
        'PredictedLogRevenue': float
    }
}
