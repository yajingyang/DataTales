import yfinance as yf
import requests
import lxml.html as LH
import requests
import pandas as pd
import csv
from dateutil.parser import parse
import json
import argparse
import re
import os
import numpy as np
from utils import get_entity_name_symbol_for_data_extraction
from tqdm import tqdm

import certifi
import ssl

ssl_context = ssl.create_default_context(cafile=certifi.where())
ssl_context.load_verify_locations(cafile='venv/lib/python3.10/site-packages/certifi/cacert.pem')

def read_raw_data(data_path):
    if data_path.endswith('csv'):
        data_df = pd.read_csv(data_path)
    elif data_path.endswith('xls'):
        data_df = pd.read_excel(data_path, header=2)
    else:
        raise "Invalid data file type!"
    return data_df


def process_raw_data(orig_df, entity):
    if entity['name'] == 'Lean Hog Cash Index':
        new_df = pd.DataFrame({'Name': entity['name'],
                               'Symbol': "",
                               'Date': orig_df['Date'].to_list(),
                               'Open': np.nan,
                               'High': np.nan,
                               'Low': np.nan,
                               'Close': orig_df['CME INDEX'].to_list()})
    else:
        orig_df = orig_df.rename(columns=lambda x:x.strip())
        date_col = [x for x in orig_df.columns if x.strip().lower() in ['time', 'date']][0]
        close_price_col = [x for x in orig_df.columns if x.strip().lower() in ['close', 'price', 'last']][0]
        vol_col_list = [x for x in orig_df.columns if 'vol' in x.lower()]
        if len(vol_col_list) > 0:
            vol_data = orig_df[vol_col_list[0]].to_list()
            if any([type(x) == str for x in vol_data]):
                vol_data = [int(float(x.replace('K', '')) * 1000)  if type(x) == str else x for x in vol_data]
        else:
            vol_data = np.nan
        new_df = pd.DataFrame({'Name': entity['name'],
                               'Symbol': entity['symbol'],
                               'Date': orig_df[date_col].to_list(),
                               'Open': orig_df['Open'].to_list(),
                               'High': orig_df['High'].to_list(),
                               'Low': orig_df['Low'].to_list(),
                               'Close': orig_df[close_price_col].to_list(),
                               'Volume': vol_data})
    new_df["Date"] = pd.to_datetime(new_df["Date"]).dt.strftime('%Y-%m-%d')
    new_df
    return new_df


def get_future_name():
    find_month_for_report_date_future(future_asset, date_string, expire_month)

def process_dairy_future_data(dairy_future_data_path):
    df = pd.read_csv(dairy_future_data_path)
    df['name'] = df.apply()


def get_file_path(entity):
    if entity['name'] == 'Silver':
        data_path = './data/raw/investing/Silver Futures Historical Data.csv'

    elif entity['data_source'] == 'investing_local':
        if "Second" in entity["name"]:
            data_dir = './data/raw/investing/second_month'
        elif "Third" in entity["name"]:
            data_dir = './data/raw/investing/third_month'

        if "Chicago SRW Wheat Future" in entity['name']:
            fname = "US Wheat Futures Historical Data.csv"
        elif "KC HRW Wheat Future" in entity['name']:
            fname = "Hard Red Winter Wheat Futures Historical Data.csv"
        else:
            pattern = r"((?:US )?(?:[\w\s]+)) Future \((Second|Third) Month\)"
            replacement = r"\1 Futures Historical Data.csv"
            fname = re.sub(pattern, replacement, entity['name'], flags=re.IGNORECASE)
            if any([us_agriculture in entity['name'] for us_agriculture in ['Corn', 'Soybean']]): fname = 'US ' + fname

        data_path = os.path.join(data_dir, fname)

    elif entity['data_source'] == 'wsj_local':
        data_dir = './data/raw/wsj/'
        pattern = r"United States (\d+)-Year Bond Yield"
        replacement = r"us_\1_year_bond_yield.csv"
        fname = re.sub(pattern, replacement, entity['name'], flags=re.IGNORECASE)
        data_path = os.path.join(data_dir, fname)

    elif entity['data_source'] == 'barchart_local' and entity['name'] == 'Feeder Cattle Cash Index' :
        data_path = './data/raw/barchart_feeder_cattle.csv'

    elif entity['data_source'] == 'barchart_local' and 'Milk' in entity['name']:
        process_dairy_future_data('./data/raw/barchart_dairy_futures.csv', entity)
        data_path = entity['name'] + '.csv'

    elif entity['name'] == 'Lean Hog Cash Index' and entity['data_source'] == 'cme_local':
        data_path = './data/raw/cme/cme_lean_hog.xls'

    else:
        raise f"Invalid entity {entity['name']}"

    return data_path


def process_investing_futures_file_name(fdir, month_name:str):
    if not os.path.exists(fdir):
        print(f"Directory {fdir} does not exist.")
        return

    # Iterate over all files in the directory
    for filename in os.listdir(fdir):
        # Check if the file is a CSV
        if filename.endswith(".csv"):
            # Construct the new filename
            new_filename = filename.replace(".csv", f" {month_name}.csv")

            # Construct full file paths
            old_file = os.path.join(fdir, filename)
            new_file = os.path.join(fdir, new_filename)

            # Rename the file
            os.rename(old_file, new_file)
            print(f"Renamed: {filename} -> {new_filename}")


def process_downloaded_data(entity):
    data_path = get_file_path(entity)
    data_df = read_raw_data(data_path)
    data_df = process_raw_data(data_df, entity)
    data_df.to_csv(output_path, mode='a', header=not os.path.exists(output_path), index=False)


def request_yfinance_data(entity):
    name_symbol_list  = get_entity_name_symbol_for_data_extraction(entity)
    for name, symbol in name_symbol_list:
        ticker = yf.Ticker(symbol)
        data_df = ticker.history(start=args.start_date, end=args.end_date)
        data_df = data_df.reset_index()
        data_df['Name'] = name
        data_df['Symbol'] = symbol
        data_df = data_df[['Name', 'Symbol', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        data_df[['Open', 'High', 'Low', 'Close', 'Volume']] = data_df[['Open', 'High', 'Low', 'Close', 'Volume']]
        data_df["Date"] = pd.to_datetime(data_df["Date"]).dt.strftime('%Y-%m-%d')
        data_df.to_csv(output_path, mode='a', header=not os.path.exists(output_path), index=False)


def drop_duplicate_dict_characterized_by_key(orig_list, key_name):
    key_values = set()
    no_dup_list = []
    for x in orig_list:
        if x[key_name] not in key_values:
            no_dup_list.append(x)
            key_values.add(x[key_name])
    return no_dup_list


def prepare_tabular_data():
    table_data_source_ref_path = "./data/financial_instrument_reference.json"
    with open(table_data_source_ref_path, 'r') as f:
        table_data_source_ref_list = json.load(f)
    table_data_source_ref_list_no_dup = drop_duplicate_dict_characterized_by_key(table_data_source_ref_list, 'name')

    for entity in tqdm(table_data_source_ref_list_no_dup):
        if 'local' in entity['data_source']:
            process_downloaded_data(entity)
        elif entity['data_source'] == 'yahoo':
            request_yfinance_data(entity)
        else:
            raise "Invalid source"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_date", default='2019-01-01', help="First date of the table data")
    parser.add_argument("--end_date", default='2023-04-30', help="Last date of the table data")
    args = parser.parse_args()

    output_path = 'data/all_table_data.csv'
    prepare_tabular_data()
