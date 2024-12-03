# Market Report Generation Dataset

This repository contains code for constructing and evaluating a market report generation dataset. The pipeline processes financial market data from various sources and pairs it with market reports to create training datasets for language models.

## 1. Dataset Construction Pipeline

### 1.1 Manual Data Collection

Due to copyright policies, some data needs to be collected manually:

#### U.S. Treasury Yields
Download from Wall Street Journal (2018/01/01 - 2023/06/30) to `data/raw/wsj/`:

| Bond Yield | URL | File Name |
|------------|-----|-----------|
| 1-Year | [Link](https://www.wsj.com/market-data/quotes/bond/BX/TMUBMUSD01Y/historical-prices) | us_1_year_bond_yield.csv |
| 2-Year | [Link](https://www.wsj.com/market-data/quotes/bond/BX/TMUBMUSD02Y/historical-prices) | us_2_year_bond_yield.csv |
| 3-Year | [Link](https://www.wsj.com/market-data/quotes/bond/BX/TMUBMUSD03Y/historical-prices) | us_3_year_bond_yield.csv |
| 5-Year | [Link](https://www.wsj.com/market-data/quotes/bond/BX/TMUBMUSD05Y/historical-prices) | us_5_year_bond_yield.csv |
| 7-Year | [Link](https://www.wsj.com/market-data/quotes/bond/BX/TMUBMUSD07Y/historical-prices) | us_7_year_bond_yield.csv |
| 10-Year | [Link](https://www.wsj.com/market-data/quotes/bond/BX/TMUBMUSD10Y/historical-prices) | us_10_year_bond_yield.csv |
| 30-Year | [Link](https://www.wsj.com/market-data/quotes/bond/BX/TMUBMUSD30Y/historical-prices) | us_30_year_bond_yield.csv |

#### Feeder Cattle Index
Download from [LRP Advisors](https://lrpadvisors.com/cme/) (use "ALL" button and "CSV" export) to `data/raw/cme/feeder_cattle_index.csv`

### 1.2 Automated Data Collection and Processing

The pipeline consists of four main scripts that process data sequentially:

#### Step 1: Download Market Data (`download_data.py`)
```bash
# Setup DataBento API key first
export DATABENTO_API_KEY=your_key_here

# Download market data
python download_data.py
```

**Objectives:**
- Download market data for individual tickers across specified timespan
- Process data from multiple sources (Yahoo Finance, CME, etc.)
- Combine all market data into a single processed file

**Output Structure:**
```
data/
├── table_data/
│   └── raw/
│       └── <market>/
│           └── <ticker>.csv    # Individual ticker data
└── intermediate/
    └── processed_data.csv      # Combined market data
```

#### Step 2: Process Table Data (`process_table_data.py`)
```bash
python process_table_data.py
```

**Objectives:**
- Extract historical data for specified time spans
- Process and format data for each market report
- Organize data by market and data source

**Output Structure:**
```
data/
└── table_data/
    └── report_table_data/
        └── <historical_time_span>/
            └── <split>/
                └── <market-report_data_source>/
                    └── <report_date>.csv
```

#### Step 3: Construct Dataset (`construct_dataset.py`)
```bash
python construct_dataset.py
```

**Objectives:**
- Combine table data with corresponding market reports
- Format prompts for model training
- Include relevant metadata

**Output Structure:**
```
data/
└── processed_dataset/
    └── <historical_time_span>/
        └── <split>.json        # Contains tables, prompts, reports, and metadata
```

#### Step 4: Tokenize Dataset (`tokenize_dataset.py`)
```bash
python tokenize_dataset.py
```

**Objectives:**
- Convert processed dataset into tokenized format
- Prepare data for training open-source language models
- Support multiple tokenizer options

**Output Structure:**
```
data/
└── tokenized_dataset/
    └── <historical_time_span>/
        └── <tokenizer>/
            └── <split>/        # Tokenized data ready for training
```
