import requests
import zipfile
import io
import os


def download_file(url, filename):
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"Successfully downloaded: {filename}")
    else:
        print(f"Failed to download: {filename}")


def download_and_unzip(url, extract_to):
    response = requests.get(url)
    if response.status_code == 200:
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Successfully downloaded and extracted to: {extract_to}")
    else:
        print(f"Failed to download: {url}")


def main():
    # Create data/raw directory if it doesn't exist
    os.makedirs('data/raw', exist_ok=True)

    # 1. Download Lean Hog Cash Index data
    lean_hog_url = "https://www.cmegroup.com/ftp/cash_settled_commodity_index_prices/historical_data/LHindx.ZIP"
    download_and_unzip(lean_hog_url, 'data/raw/unzip')

    # Rename the extracted file to cme_lean_hog.xls
    extracted_files = os.listdir('data/raw/unzip')
    for file in extracted_files:
        if file.endswith('.xls'):
            os.rename(f'data/raw/unzip/{file}', 'data/raw/cme_lean_hog.xls')
            break

    # # 2. Download cheese data
    # cheese_url = "https://www.cheesemarketnews.com/marketarchive/images/cheese.xls"
    # download_file(cheese_url, 'data/raw/cheese.xls')
    #
    # # 3. Download butter data
    # butter_url = "https://www.cheesemarketnews.com/marketarchive/images/butter.xls"
    # download_file(butter_url, 'data/raw/butter.xls')


if __name__ == "__main__":
    main()