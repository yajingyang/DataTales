# Data Preparation Guide

This guide outlines the steps to prepare input tabular data from various sources. Due to copyright policies, we cannot redistribute the data directly. Please follow these steps to gather the required information.

## 1. Run Download Script

Execute the following script to download and process some of the data:

```
python download_data_from_url.py
```

This script will download **Lean Hog Cash Index** data from [CME](https://www.cmegroup.com/ftp/cash_settled_commodity_index_prices/historical_data/LHindx.ZIP), unzip the file, and store it as `data\raw\cme_lean_hog.xls`.

[//]: # (2. Download [cheese]&#40;https://www.cheesemarketnews.com/marketarchive/images/cheese.xls&#41; and [butter]&#40;https://www.cheesemarketnews.com/marketarchive/images/butter.xls&#41; data.)

## 2. U.S. Treasury Yields

Download U.S. Treasury yields from the Wall Street Journal for the period 01/01/2019 to 06/30/2023 and save as the filename listed:

| Bond Yield | URL | File Name |
|------------|-----|-----------|
| 1-Year | [Link](https://www.wsj.com/market-data/quotes/bond/BX/TMUBMUSD01Y/historical-prices) | us_1_year_bond_yield.csv |
| 2-Year | [Link](https://www.wsj.com/market-data/quotes/bond/BX/TMUBMUSD02Y/historical-prices) | us_2_year_bond_yield.csv |
| 3-Year | [Link](https://www.wsj.com/market-data/quotes/bond/BX/TMUBMUSD03Y/historical-prices) | us_3_year_bond_yield.csv |
| 5-Year | [Link](https://www.wsj.com/market-data/quotes/bond/BX/TMUBMUSD05Y/historical-prices) | us_5_year_bond_yield.csv |
| 7-Year | [Link](https://www.wsj.com/market-data/quotes/bond/BX/TMUBMUSD07Y/historical-prices) | us_7_year_bond_yield.csv |
| 10-Year | [Link](https://www.wsj.com/market-data/quotes/bond/BX/TMUBMUSD10Y/historical-prices) | us_10_year_bond_yield.csv |
| 30-Year | [Link](https://www.wsj.com/market-data/quotes/bond/BX/TMUBMUSD30Y/historical-prices) | us_30_year_bond_yield.csv |

## 3. Feeder Cattle Cash Index

Download **Feeder Cattle Cash Index** data from [Barchart](https://www.barchart.com/futures/quotes/GFY00/price-history/historical) and store it as `data\raw\barchart_feeder_cattle.csv`.

Note: A free account provides up to 2 years of historical data. For longer timespan required for fine-tuning with the train set, a paid account (14-day free trial available) is necessary.

## 4. Silver Price
Download **Silver** data from Investing.com ([Link](https://www.investing.com/commodities/silver-historical-data)).

## 4. Agriculture Commodity Futures

Download the following agriculture commodity second/third month futures from Investing.com:

| Commodity | Second Month | Third Month |
|-----------|--------------|-------------|
| Live Cattle | [Link](https://www.investing.com/commodities/live-cattle-historical-data?cid=1178211) | [Link](https://www.investing.com/commodities/live-cattle-historical-data?cid=1178212) |
| Feeder Cattle | [Link](https://www.investing.com/commodities/feed-cattle-historical-data?cid=1178353) | [Link](https://www.investing.com/commodities/feed-cattle-historical-data?cid=1178354) |
| Corn Future | [Link](https://www.investing.com/commodities/us-corn-historical-data?cid=1178335) | [Link](https://www.investing.com/commodities/us-corn-historical-data?cid=1178336) |
| Lean Hog Future | [Link](https://www.investing.com/commodities/lean-hogs-historical-data?cid=1178213) | [Link](https://www.investing.com/commodities/lean-hogs-historical-data?cid=1178214) |
| Soybean Future | [Link](https://www.investing.com/commodities/us-soybeans-historical-data?cid=1178326) | [Link](https://www.investing.com/commodities/us-soybeans-historical-data?cid=1178327) |
| Soybean Oil Future | [Link](https://www.investing.com/commodities/us-soybean-oil-historical-data?cid=1178329) | [Link](https://www.investing.com/commodities/us-soybean-oil-historical-data?cid=1178330) |
| Soybean Meal Future | [Link](https://www.investing.com/commodities/us-soybean-meal-historical-data?cid=1178332) | [Link](https://www.investing.com/commodities/us-soybean-meal-historical-data?cid=1178333) |
| Chicago SRW Wheat Future | [Link](https://www.investing.com/commodities/us-wheat-historical-data?cid=1178338) | [Link](https://www.investing.com/commodities/us-wheat-historical-data?cid=1178339) |
| KC HRW Wheat Future | [Link](https://www.investing.com/commodities/hard-red-winter-wheat-historical-data?cid=1178217) | [Link](https://www.investing.com/commodities/hard-red-winter-wheat-historical-data?cid=1178218) |

## 5. Milk Futures

Download **Class III Milk Future (Second Month)** and **Class IV Milk Future (Second Month)** from [Barchart](https://www.barchart.com/excel?ref=histDownload). 

Note: A free account allows only one download per day. Consider signing up for Barchart for Excel and installing the plug-in to download data in an Excel spreadsheet. Alternatively, you may skip the daily market reports (283 reports) by removing the corresponding rows from the entity reference file `data/financial_instrument_reference.json`.

Manually download the historical data for the following tickers and store in a CSV file named `milk_second_month_future.csv`:

### Class III Milk Future (Second Month)
- DLZ21, DLF22, DLG22, DLH22, DLJ22, DLK22, DLM22, DLN22, DLQ22, DLU22, DLV22, DLX22, DLZ22, DLF23, DLG23, DLH23, DLJ23, DLK23, DLM23, DLN23, DLQ23

### Class IV Milk Future (Second Month)
- DKF22, DKG22, DKH22, DKJ22, DKK22, DKM22, DKN22, DKQ22, DKU22, DKV22, DKX22, DKZ22, DKF23, DKG23, DKH23, DKJ23, DKK23, DKM23, DKN23, DKQ23

## Data Sources

The tabular data is extracted from the following sources:
- [Yahoo! Finance](https://finance.yahoo.com/): All others not listed below
- [Investing.com](https://www.investing.com/): U.S. commodity futures and silver price
- [WSJ](https://www.wsj.com/): U.S. Treasury yields
- [CME](https://www.cmegroup.com/): Lean Hog Cash Index
- [Cheese Market News](https://www.cheesemarketnews.com/): Cheese and butter prices
- [Barchart](https://www.barchart.com/): Feeder Cattle Cash Index, Class III/IV milk futures