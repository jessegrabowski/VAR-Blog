import os
import json
import requests
import pandas as pd
from yfinance import Tickers
from pandas_datareader.fred import FredReader


def BISScraper(symbols):
    if isinstance(symbols, str):
        symbols = [symbols]
    
    headers = {'Content-type': 'application/json'}
    start_year = 1950
    end_year = 2030
    years = range(start_year, end_year, 10)
    all_data = []
    for start in years:
        end = start + 10
        data = json.dumps({"seriesid": symbols,"startyear": f"{start}", "endyear":f"{end}"})
        p = requests.post('https://api.bls.gov/publicAPI/v1/timeseries/data/', data=data, headers=headers)
        json_data = json.loads(p.text)
        all_data.append(json_data)
    
    all_series = [x['Results']['series'] for x in all_data]    
    chunk_dfs = []
    
    for time_chunk in range(len(all_series)):
        chunk = all_series[time_chunk]
        for series in chunk:
            series_id = series['seriesID']
            chunk_df = pd.DataFrame(series['data'])
            if len(chunk_df) > 0:
                chunk_df['series_id'] = series_id
                chunk_dfs.append(chunk_df)
    
    bis_df = (pd.concat(chunk_dfs)
              .assign(Date = lambda x: pd.to_datetime(x.year + x.period.str.replace('0', '')),
                      value = lambda x: x.value.apply(float))
              .sort_values(by='Date')
              .loc[:, ['Date', 'value', 'series_id']]
              .pivot_table(index='Date', columns='series_id', values='value'))
    
    bis_df.index.name = None
    bis_df.columns.name = None
    
    return bis_df

def download_raw_data(variables, source='fred', out_path='data/', fname='VAR_data_raw.csv', force_update = False):
    if isinstance(variables, str):
        variables = [variables]
    
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    
    f_path = os.path.join(out_path, fname)
    df = None
    new_variables = variables.copy()
    vars_to_update = []
    
    if os.path.isfile(f_path):
        df = pd.read_csv(f_path, index_col=0, parse_dates=[0], infer_datetime_format=True)
        cols = df.columns
        new_variables = [x for x in variables if x not in cols]
        vars_to_update = [x for x in variables if x in cols]
        
    if len(new_variables) == 0 and not force_update:
        print('All requested variables have been downloaded')
        return
    
    to_download = new_variables if not force_update else variables
    
    if source.lower() == 'fred':
        new_df = FredReader(symbols=to_download, start='1900-01-01', end=None).read()
    
    if source.lower() == 'yahoo':
        yf_ticker = Tickers(to_download)
        new_df = yf_ticker.download(period='max', progress=False)['Close']
    
    if source.lower() == 'bis':
        new_df = BISScraper(symbols=to_download)
    
    # Update columns that are already present
    if force_update:
        ret_df = df.copy()
        ret_df[vars_to_update] = new_df[vars_to_update]
        ret_df = ret_df.join(new_df.drop(columns=vars_to_update))
       
    else:
        ret_df = df.join(new_df, how='outer') if df is not None else new_df
    
    ret_df.to_csv(f_path)