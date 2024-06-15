
# Data Preparation

### 1. Extract tabular data
The tabular data is extracted from the following sources:
- *[Yahoo! Finance](https://finance.yahoo.com/)*: all others not listed below
- *[Investing.com](https://www.investing.com/)*: U.S. treasury yield and silver price
- *[CME](https://www.cmegroup.com/)*: Lean Hog Cash Index
- *[Barchart](https://www.barchart.com/)*: Feeder Cattle Cash Index

To prepare the input tabular data:


1. Download the U.S. treasury yield and silver price data from Investing.com. Make sure to change the start and end date
to cover the first and last date of the report and the required history timespan for the first date. E.g., if the reports
date range from 1/1/2022 - 31/12/2022 and the tabular data timespan is 1 month, then the start and end date should be set
to 1/12/2021-31/12/2022. Store the data files under directory `data\raw\investing`

   - ***Silver***: https://www.investing.com/commodities/silver-historical-data
   - ***United States 1-Year Bond Yield***: https://www.investing.com/rates-bonds/u.s.-1-year-bond-yield-historical-data
   - ***United States 2-Year Bond Yield***: https://www.investing.com/rates-bonds/u.s.-2-year-bond-yield-historical-data
   - ***United States 3-Year Bond Yield***: https://www.investing.com/rates-bonds/u.s.-3-year-bond-yield-historical-data
   - ***United States 5-Year Bond Yield***: https://www.investing.com/rates-bonds/u.s.-5-year-bond-yield-historical-data
   - ***United States 7-Year Bond Yield***: https://www.investing.com/rates-bonds/u.s.-7-year-bond-yield-historical-data
   - ***United States 10-Year Bond Yield***: https://www.investing.com/rates-bonds/u.s.-10-year-bond-yield-historical-data
   - ***United States 30-Year Bond Yield***: https://www.investing.com/rates-bonds/u.s.-30-year-bond-yield-historical-data

2. Download ***Lean Hog Cash Index*** data from CME: 
https://www.cmegroup.com/ftp/cash_settled_commodity_index_prices/historical_data/LHindx.ZIP, unzip the file and store as `data\raw\cme_lean_hog.xls`.
3. Download ***Feeder Cattle Cash Index*** data from Barchart: https://www.barchart.com/futures/quotes/GFY00/price-history/historical. 
Store the file as `data\raw\barchart_feeder_cattle.csv`.\
   Noted that Barchart provides the historical data up to 2 years prior to the day of download with free account. 
   This is sufficient for evaluation with the test set, but the fine-tuning with train set will require longer timespan
   hence a paid account (free-trail of 14 days offered) is required.

The Yahoo! Finance data will be extracted through API request on-the-fly during dataset construction.

4. Process raw data and write into local csv file: 

      python construct_dataset.py



### 2. Construct dataset:
Pair the report with the corresponding tabular data and output as json file. The constructed data includes three subsets: 
train/test/validation. 

      python construct_dataset.py



### 3. Process dataset


# Model fine-tuning (optional)
### 1. 

# Model evaluation
### 1. Model Inference
### 2. Automatic metrics (BLEU, ROUGE)
### 3. Accuracy evaluation
