import pandas as pd
import json
from datetime import datetime, timedelta
import os
from dataclasses import dataclass
from typing import List, Dict, Optional
from pathlib import Path

@dataclass
class MarketReport:
    source: str
    market: str
    date: datetime
    passage: str

@dataclass
class FuturesContract:
    symbol: str
    expiration_date: datetime
    base_symbol: str

class DataLoader:
    @staticmethod
    def load_json(file_path: str) -> Dict:
        with open(file_path, 'r') as f:
            return json.load(f)

    @staticmethod
    def load_csv(file_path: str, parse_dates: List[str], **kwargs) -> pd.DataFrame:
        return pd.read_csv(file_path, parse_dates=parse_dates, **kwargs)

class FuturesDataProcessor:
    def __init__(self, futures_data: pd.DataFrame):
        self.futures_data = futures_data

    def get_active_contracts(self, base_symbol: str, date: datetime, n: int = 3) -> pd.DataFrame:
        return self.futures_data[
            (self.futures_data['symbol'].str.startswith(base_symbol)) &
            (self.futures_data['expiration_date'] > date)
        ].sort_values('expiration_date').head(n)

    @staticmethod
    def create_product_name(base_name: str, rank: int, expiration_date: datetime) -> str:
        month_name = expiration_date.strftime("%B")
        rank_map = {
            1: "front month",
            2: "second month",
            3: "third month"
        }
        return f"{base_name} ({rank_map[rank]}) ({month_name})"

class MarketDataExtractor:
    def __init__(self, processed_data: pd.DataFrame, reference_data: List[Dict], 
                 futures_processor: FuturesDataProcessor):
        self.processed_data = processed_data
        self.reference_data = reference_data
        self.futures_processor = futures_processor

    def _extract_futures_data(self, item: Dict, start_date: datetime, 
                            end_date: datetime) -> pd.DataFrame:
        market_data = pd.DataFrame()
        base_symbol = item['symbol']
        
        # Get active contracts for the report date (end_date)
        active_futures = self.futures_processor.get_active_contracts(base_symbol, end_date)
        active_symbols = active_futures['symbol'].tolist()
        
        if active_symbols:
            # Filter processed data for the active contract symbols and date range
            market_data = self.processed_data[
                (self.processed_data['Symbol'].isin(active_symbols)) &
                (self.processed_data['Date'] >= start_date) &
                (self.processed_data['Date'] <= end_date)
            ].copy()
            
            # Add rank and product name information
            for rank, (_, future) in enumerate(active_futures.iterrows(), 1):
                mask = market_data['Symbol'] == future['symbol']
                market_data.loc[mask, 'Product Name'] = self.futures_processor.create_product_name(
                    item['name'], rank, future['expiration_date']
                )
        
        return market_data.sort_values(['Product Name', 'Date']).reset_index(drop=True)

    def _extract_symbol_data(self, item: Dict, start_date: datetime, 
                           end_date: datetime) -> pd.DataFrame:
        return self.processed_data[
            (self.processed_data['Symbol'] == item['symbol']) &
            (self.processed_data['Date'] >= start_date) &
            (self.processed_data['Date'] <= end_date)
        ]

    def _extract_product_data(self, item: Dict, start_date: datetime, 
                            end_date: datetime) -> pd.DataFrame:
        return self.processed_data[
            (self.processed_data['Product Name'] == item['name']) &
            (self.processed_data['Date'] >= start_date) &
            (self.processed_data['Date'] <= end_date)
        ]
 
    def extract_market_data(self, market: str, start_date: datetime, 
                          end_date: datetime) -> pd.DataFrame:
        market_items = [item for item in self.reference_data if item['market'] == market]
        market_data = pd.DataFrame()

        for item in market_items:
            data = pd.DataFrame()
            
            if item['data_source'] == 'cme' and item['symbol']:
                data = self._extract_futures_data(item, start_date, end_date)
            elif item['symbol']:
                data = self._extract_symbol_data(item, start_date, end_date)
            else:
                data = self._extract_product_data(item, start_date, end_date)
                
            market_data = pd.concat([market_data, data])
        
        market_data = market_data.drop_duplicates(['Date', 'Product Name', 'Symbol'])

        return market_data

class MarketDataManager:
    def __init__(self, config: Dict[str, str]):
        self.config = config
        self.data_loader = DataLoader()
        self._initialize_data()

    def _initialize_data(self):
        self.reference_data = self.data_loader.load_json(self.config['reference_data_path'])
        self.processed_data = self.data_loader.load_csv(
            self.config['processed_data_path'], 
            parse_dates=['Date']
        )
        report_data = self.data_loader.load_csv(
            self.config['report_data_path'], 
            parse_dates=['date'],
            sep='\t',
            encoding='utf-16'
        )
        
        dataset_split_reference = self.data_loader.load_csv(
            self.config['split_ref_path'],
            parse_dates=['date']
        )
        self.report_data = pd.merge(report_data, dataset_split_reference, on=['source', 'market', 'date'])
        futures_data = self.data_loader.load_csv(
            self.config['futures_data_path'],
            parse_dates=['expiration_date']
        )
        self.futures_processor = FuturesDataProcessor(futures_data)
        self.market_extractor = MarketDataExtractor(
            self.processed_data,
            self.reference_data,
            self.futures_processor
        )

    def process_reports(self, history_span_in_days: int):
        for _, report in self.report_data.iterrows():
            market = report['market']
            source = report['source']
            report_date = report['date']
            split = report['split']
            start_date = report_date - timedelta(days=history_span_in_days)
            
            self._process_single_report(market, source, report_date, start_date, split)

    def _process_single_report(self, market: str, source, report_date: datetime, start_date: datetime, split: str):
        output_dir = Path(self.config['output_base_path']) / split / f"{market.replace(' ', '_')}-{source}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        market_data = self.market_extractor.extract_market_data(
            market, start_date, report_date
        )
        
        if not market_data.empty:
            output_path = output_dir / f"{report_date.strftime('%Y-%m-%d')}.csv"
            market_data.to_csv(output_path, index=False)
            print(f"Data extracted for {market} on {report_date.strftime('%Y-%m-%d')} "
                  f"saved to {output_path}")
        else:
            print(f"No data found for {market} on {report_date.strftime('%Y-%m-%d')}")

def main():
    history_time_span = '1week'
    history_time_span_map = {
        '1day': 1,
        '1week': 7,
        '1month': 30,
        '1year': 365,
        '2year': 365*2
    }
    config = {
        'reference_data_path': 'data/references/financial_instrument_reference.json',
        'processed_data_path': 'data/table_data/intermediate/processed_data.csv', 
        'report_data_path': 'data/reports/reports.tsv',
        'split_ref_path': 'data/references/split_ref.csv',
        'futures_data_path': 'data/references/futures_symbol.csv',
        'output_base_path': f'data/table_data/report_table_data/{history_time_span}'
    }

    manager = MarketDataManager(config)
    manager.process_reports(history_span_in_days=history_time_span_map[history_time_span])

if __name__ == "__main__":

    main()