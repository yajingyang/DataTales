import re
import datetime

future_month_map = {
    "Live Cattle Future": "February, April, June, August, October, December",
    "Feeder Cattle Future": "January, March, April, May, August, September, October, November",
    "Corn Future": "March, May, July, September, December",
    "Class III Milk Future": "January, February, March, April, May, June, July, August, September, October, November, December",
    "Class IV Milk Future": "January, February, March, April, May, June, July, August, September, October, November, December",
    "Lean Hog Future": "February, April, May, June, July, August, October, December",
    "Soybean Future": "January, March, May, July, August, September, October, December",
    "Soybean Meal Future": "January, March, May, July, August, September, October, December",
    "Soybean Oil Future": "January, March, May, July, August, September, October, December",
    "KC HRW Wheat Future": "March, May, July, September, December",
    "Chicago SRW Wheat Future": "March, May, July, September, December",
}

def get_future_symbol(asset_code, future_month, future_year):
    month_code_map = {
        "January": "F",
        "February": "G",
        "March": "H",
        "April": "J",
        "May": "K",
        "June": "M",
        "July": "N",
        "August": "Q",
        "September": "U",
        "October": "V",
        "November": "X",
        "December": "Z"
    }
    future_symbol = asset_code.replace('{month}', month_code_map[future_month]).replace('{year}', future_year)
    return future_symbol




def find_month_for_report_date_future(future_asset, date_string, expire_month):
    expire_month_list = ['front month', 'second month', 'third month']
    assert expire_month in expire_month_list, "Invalid expiring month! "
    month_map = {month: index for index, month in enumerate(calendar.month_name) if month}

    month_list = future_month_map[future_asset].split(',')
    month_list = [x.strip() for x in month_list]
    month_num_list = [month_map[mn] for mn in month_list]
    month_number = int(datetime.strptime(date_string, "%Y-%m-%d").strftime("%m"))
    future_year_name = datetime.strptime(date_string, "%Y-%m-%d").strftime("%y")

    if expire_month is not None:
        try:
            coming_month_num = [x for x in month_num_list if x - month_number >= 0]
            expire_front_month_num = coming_month_num[0] if len(coming_month_num) > 0 else month_num_list[0]
        except:
            print()
        expire_month_index = month_num_list.index(expire_front_month_num) + expire_month_list.index(expire_month)
        if expire_month_index >= len(month_list):
            expire_month_index = expire_month_index - len(month_list)
            future_year_name = str(int(future_year_name) + 1)
        future_month_name = month_list[expire_month_index]
    else:
        future_month_name = month_list[expire_month_index]
    return future_year_name, future_month_name


def find_month_year_list_for_future_data_extraction(future_asset, start_year, end_year):
    month_list = future_month_map[future_asset].split(',')
    month_list = [x.strip() for x in month_list]

    year_month_list = [(y, m) for y in range(start_year, end_year+1) for m in month_list]
    return year_month_list


def get_entity_name_symbol_for_data_extraction(row):
    name_symbol_list = []

    match_future = re.match("(.*) \((.*)\)", row['name'])
    if match_future:
        asset_name, expire_month = match_future.group(1), match_future.group(2)
        future_year_month_list = find_month_year_list_for_future_data_extraction(asset_name, 19, 23)
        for future_year_name, future_month_name in future_year_month_list:
            asset_code = row['symbol'].strip()
            symbol = get_future_symbol(asset_code, future_month_name, str(future_year_name))
            name = f"{asset_name} ({future_month_name} {future_year_name})"
            name_symbol_list.append((name, symbol))
    else:
        name = row['name'].strip()
        symbol = row['symbol']
        name_symbol_list = [(name, symbol)]
    return name_symbol_list
