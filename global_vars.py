FRED_VARS = {'GDP':'GDP',
             'CPI':'CPIAUCSL',
             'Federal Funds Rate':'FEDFUNDS',
             'Commodity price index':'PPIACO',
             'Depository instituional reserves: nonborrowed ':'NONBORRES',
             'Depository inst reserves: total': 'TOTRESNS',
             'Money stock: M2':'M2SL',
             'Real personal expenditures (quantity)':'DPCERA3M086SBEA',
             'Industrial Production':'INDPRO',
             'Capacity Utilization':'CUMFNS',
             'U5 Unemployment':'U5RATE',
             'Housing Starts':'HOUST',
             'PPI: Finished Goods': 'WPSFD49207',
             'Personal Consumption price index':'PCEPI',
             'Money stock: M1':'M1SL',
             '10-Year bond yield':'GS10',
             'Real Effective Exchange Rate':'RNUSBIS',
             'Total employees: Non-farm':'PAYEMS',
             '10-Y minus 3-M treasure spread':'T10Y3MM',
             'UM Consumer Sentiment':'UMCSENT',
             'Real gross private investment':'GPDIC1',
             'Hourly non-farm labor productivity':'OPHNFB',
             'Real imports':'IMPGSC1',
             'Real exports':'EXPGSC1',
             'Real government expenditure':'W068RCQ027SBEA',
             'Hours worked for all non-farm':'HOANBS',
             'Total nonrevolving consumer credit outstanding':'NONREVSL',
             'Commerical and industrial loans outstanding':'BUSLOANS',
             'Crude oil price':'MCOILWTICO'}

YAHOO_VARS = {'SP500':'^GSPC',
              'GJIA':'^DJI'}

# The BIS public API is mega stingy, so don't call it too much
BIS_VARS = {'Median usual weekly earnings':'LES1252881600'}

COL_NAMES = {
    'GDP': 'GDPGrowth',
    'CPIAUCSL': 'CPI',
    'FEDFUNDS': 'FedFunds',
    'PPIACO': 'PPI_All',
    'NONBORRES': 'NonBorrowReserves',
    'TOTRESNS': 'TotalReserves',
    'M2SL': 'M2Supply',
    'DPCERA3M086SBEA': 'PersonalConsum',
    'INDPRO': 'IndustrialProd',
    'CUMFNS': 'CapacityUtil',
    'U5RATE': 'U5Unemp',
    'HOUST': 'HousingStarts',
    'WPSFD49207': 'PPI_Finished',
    'PCEPI': 'PersonalConsumPrice',
    'M1SL': 'M1Supply',
    'GS10': '10YBond',
    'RNUSBIS': 'ExRate',
    'PAYEMS': 'TotalEmployees',
    'T10Y3MM': 'TSpread',
    'UMCSENT': 'ConsumerSent',
    'GPDIC1': 'PrivInvestment',
    'OPHNFB': 'HourlyProductivity',
    'IMPGSC1': 'Imports',
    'EXPGSC1': 'Exports',
    'W068RCQ027SBEA': 'GovtSpending',
    'HOANBS': 'HoursWorked',
    'NONREVSL': 'ConsumerCredit',
    'BUSLOANS': 'BusinessLoans',
    'MCOILWTICO': 'CrudeOil',
    '^GSPC': 'S&P500',
    '^DJI':'DJIA',
    'LES1252881600':'WeeklyEarnings'
}

VARIABLE_TYPES = {
    'GDPGrowth': 'flow',
    'CPI': 'stock', #All indices are stocks
    'FedFunds': 'stock', #Prices are stocks
    'PPI_All': 'stock', # indices are stocks
    'NonBorrowReserves': 'stock', #money supply is a stock
    'TotalReserves': 'stock', # money supply
    'M2Supply': 'stock', # money supply
    'PersonalConsum': 'flow', # gdp component: flow
    'IndustrialProd': 'stock', # index 
    'CapacityUtil': 'stock', # percentag
    'U5Unemp': 'stock', # index
    'HousingStarts': 'flow', # starts/month: flow
    'PPI_Finished': 'stock', #index
    'PersonalConsumPrice': 'stock', #price
    'M1Supply': 'stock', # money supply
    '10YBond': 'stock', # price
    'ExRate': 'stock', # price
    'TotalEmployees': 'flow', # thousand people/month -> flow
    'TSpread': 'stock', # price
    'ConsumerSent': 'stock', # index
    'PrivInvestment': 'flow', # gdp component
    'HourlyProductivity': 'stock', # index
    'Imports': 'flow',  # gdp component
    'Exports': 'flow',  # gdp component
    'GovtSpending': 'flow', # gdp component
    'HoursWorked': 'stock', # index
    'ConsumerCredit': 'stock', # cumulative quantity
    'BusinessLoans': 'stock', #cumulative quantity
    'CrudeOil': 'stock', #price
    'S&P500': 'stock', #price
    'DJIA': 'stock', #price
    'WeeklyEarnings': 'stock' #price
}

PREPROCESSING_CODES = {
    1: 'i', # IdentityTransform
    2: 'd', # DifferenceTransform
    3: 'dd', #Diff + Diff
    4: 'l',  # LogTransform
    5: 'ld', # Log + Diff
    6: 'ldd'} # Log + Diff + Diff

PREPROCESS_BY_COL = {
    'GDPGrowth': 5,
    'CPI': 6,
    'FedFunds': 2,
    'PPI_All': 5,
    'NonBorrowReserves': 3,
    'TotalReserves': 6,
    'M2Supply': 6,
    'PersonalConsum': 5,
    'IndustrialProd': 6,
    'CapacityUtil': 1,
    'U5Unemp': 2,
    'HousingStarts': 4,
    'PPI_Finished': 6,
    'PersonalConsumPrice': 6,
    'M1Supply': 6,
    '10YBond': 2,
    'ExRate': 5,
    'TotalEmployees': 5,
    'TSpread': 1,
    'ConsumerSent': 2,
    'PrivInvestment': 1,
    'HourlyProductivity': 5,
    'Imports': 5,
    'Exports': 5,
    'GovtSpending': 5,
    'HoursWorked': 5,
    'ConsumerCredit': 6, 
    'BusinessLoans': 6,
    'CrudeOil': 5,
    'S&P500': 5,
    'DJIA': 5,
    'WeeklyEarnings': 5
}