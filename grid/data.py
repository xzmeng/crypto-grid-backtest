import os
import shutil
from os.path import dirname
from typing import List, Tuple

import pandas as pd
import requests
import numpy as np


BINANCE_UM_START_DATE = pd.Timestamp('2020-1-1')
DATA_DIR = os.path.join(dirname(dirname(__file__)), 'data')
PANDAS_DIR = os.path.join(DATA_DIR, 'pandas')
CSV_DIR = os.path.join(DATA_DIR, 'csv')

def download_file(url, filename):
    """
    Return True if the file is available, False otherwise
    """
    if os.path.exists(filename):
        print(f'{filename} already exists. Abort download it')
        return True
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with requests.get(url, stream=True) as r:
        if r.status_code == 404:
            print(f'{url} does not exist on https://data.binance.vision')
        elif r.status_code != 200:
            print(
                f'{filename} failed to download. status code: {r.status_code}')
        else:
            try:
                with open(filename, 'wb') as f:
                    shutil.copyfileobj(r.raw, f)
                return True
            # Remove the file if there is any error or KeyboardInterrupt
            except Exception:
                os.remove(filename)
                raise

    return False

def unzip_file(filename):
    csvname = filename.replace('.zip', '.csv')
    if os.path.exists(csvname):
        print(f'{csvname} already exists. Abort unzip the zip file')
        return
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    try:
        shutil.unpack_archive(filename, extract_dir=os.path.dirname(filename))
    # Remove the file if there is any error or KeyboardInterrupt
    except Exception:
        os.remove(filename)
        raise


def get_dates(start: pd.Timestamp, end: pd.Timestamp) -> List[Tuple[str, str]]:
    """Return a list of (freq, date) tuple"""

    # If the start date is in the same month as the end date,
    # download the daily data from start to end
    if start.replace(day=1).date() == end.replace(day=1).date():
        days = pd.date_range(start, end).strftime('%Y-%m-%d').to_list()
        dates = [('daily', day) for day in days]
        return dates
    # If the start date is not the same month as the end date,
    # first download the monthly data, then daily data in the month of end
    else:
        months = pd.date_range(start, end, freq='M').strftime('%Y-%m').to_list()
        days = pd.date_range((end + pd.Timedelta('1 day')).replace(day=1), end).strftime('%Y-%m-%d').to_list()
        dates = [('monthly', month) for month in months]
        dates += [('daily', day) for day in days]
        return dates

def download_and_unzip_csv_files(symbol, dates):
    print(f'Start downloading {symbol} data')
    for freq, date in dates:
        print(f'{freq}\t{date}\t')
    url_template = 'https://data.binance.vision/data/futures/um/{freq}/aggTrades/{symbol}/{symbol}-aggTrades-{date}.zip'
    for date in dates:
        url = url_template.format(freq=date[0], symbol=symbol, date=date[1])
        freq, date_ = date
        filename = f'{CSV_DIR}/{symbol}/{freq}/{symbol}-aggTrades-{date_}.zip'
        csvname = filename.replace('.zip', '.csv')
        if os.path.exists(csvname):
            print(f'{csvname} already exists. continue to next')
            continue
        print(f'Downloading {filename}')
        if download_file(url, filename):
            print(f'Unzipping {filename}')
            unzip_file(filename)


def remove_files(symbol, dates):
    """Remove the zip and csv files"""
    for date in dates:
        freq, date_ = date
        zipname = f'{CSV_DIR}/{symbol}/{freq}/{symbol}-aggTrades-{date_}.zip'
        csvname = zipname.replace('.zip', '.csv')
        print(f'Remove {zipname}...')
        try:
            os.remove(zipname)
        except FileNotFoundError:
            pass
        print(f'Remove {csvname}...')
        try:
            os.remove(csvname)
        except FileNotFoundError:
            pass

def load_price_dataframe_from_csv(filename):
    print(f'Loading {filename}')
    if not os.path.exists(filename):
        print(f'{filename} does not exist, pass')
        return None
    df = pd.read_csv(filename, usecols=[1, 5], names=['price', 'datetime'])
    df['datetime'] = pd.to_datetime(df.datetime, unit='ms')
    df.set_index('datetime', inplace=True)
    df = df.resample('1s').first().dropna()
    return df

def load_price_dataframe(symbol, dates):
    dfs = []
    for date in dates:
        freq, date_ = date
        filename = f'{CSV_DIR}/{symbol}/{freq}/{symbol}-aggTrades-{date_}.csv'
        df = load_price_dataframe_from_csv(filename)
        if df is not None:
            dfs.append(df)
    if dfs:
        return pd.concat(dfs)
    else:
        return None

def download_to_pickle(symbol, start, end):
    dates = get_dates(start, end)
    download_and_unzip_csv_files(symbol, dates)
    df = load_price_dataframe(symbol, dates)
    os.makedirs(PANDAS_DIR, exist_ok=True)
    df.to_pickle(f'{PANDAS_DIR}/{symbol}.pkl')
    return df


def load_data(symbol, start=None, end=None, right_only=False):
    if start is None:
        start = BINANCE_UM_START_DATE
    if end is None:
        end = pd.Timestamp.today()
    if isinstance(start, str):
        start = pd.Timestamp(start)
    if isinstance(end, str):
        end = pd.Timestamp(end)
    try:
        file = f'{PANDAS_DIR}/{symbol}.pkl'
        print(f'Reading {file}')
        df = pd.read_pickle(file)
    except FileNotFoundError:
        return download_to_pickle(symbol, start, end)
    
    df_start = df.index[0]
    df_end = df.index[-1]

    if right_only:
        left_df = None
    else:
        left_start = BINANCE_UM_START_DATE
        left_end = df_start - pd.Timedelta('1 day')
        left_dates = get_dates(left_start, left_end)
        download_and_unzip_csv_files(symbol, left_dates)
        left_df = load_price_dataframe(symbol, left_dates)

    right_start = df_end + pd.Timedelta('1 day')
    right_end = pd.Timestamp.today()
    right_dates = get_dates(right_start, right_end)
    download_and_unzip_csv_files(symbol, right_dates)
    right_df = load_price_dataframe(symbol, right_dates)
    new_df = pd.concat([left_df, df, right_df])

    if len(df) != len(new_df):
        print('drop duplicated...')
        new_df = new_df[~new_df.index.duplicated(keep='first')]
        print('drop duplicated done')

        print('old:')
        print(df)
        print('new:')
        print(new_df)

        old = f'{PANDAS_DIR}/{symbol}_prev.pkl'
        new = f'{PANDAS_DIR}/{symbol}.pkl'
        print(f'Saving old data to {old}')
        df.to_pickle(old)
        print(f'Saving new data to {new}')
        new_df.to_pickle(new)

    print('Removing zip and csv files')
    remove_files(symbol, get_dates(new_df.index[0], new_df.index[-1]))
    return new_df

def load_local_data(symbol):
    return pd.read_pickle(f'{PANDAS_DIR}/{symbol}.pkl')
