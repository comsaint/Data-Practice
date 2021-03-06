USE_COLS = ['channelGrouping', 'date', 'fullVisitorId', 'sessionId',
       'visitNumber', 'visitStartTime',
       'device.browser',
       'device.deviceCategory', 'device.isMobile',
       'device.operatingSystem',
       'geoNetwork.city',
       #'geoNetwork.continent',
       'geoNetwork.country',
       'geoNetwork.metro',
       'geoNetwork.networkDomain',
       'geoNetwork.region',
       #'geoNetwork.subContinent',
       'totals.bounces', 'totals.hits', 'totals.newVisits', 'totals.pageviews',
       'totals.transactionRevenue', 'trafficSource.adContent',
       'trafficSource.adwordsClickInfo.adNetworkType',
       #'trafficSource.adwordsClickInfo.isVideoAd',
       'trafficSource.adwordsClickInfo.page',
       'trafficSource.adwordsClickInfo.slot',
       'trafficSource.campaign',
       'trafficSource.isTrueDirect',
       'trafficSource.keyword',
       'trafficSource.medium',
       'trafficSource.referralPath',
       'trafficSource.source'
            ]

NUM_COLS = ['totals.bounces', 'totals.hits', 'totals.newVisits', 'totals.pageviews']
TARGET = ['totals.transactionRevenue']
