import os
import json
import pandas as pd
from utils import get_future_symbol, find_month_for_report_date_future
import re
from tqdm import tqdm
from datetime import datetime

def get_entity_name_symbol(row):
    match_future = re.match("(.*) \((.*)\)", row['name'])
    if match_future:
        asset_name, expire_month = match_future.group(1), match_future.group(2)
        future_year_name, future_month_name = find_month_for_report_date_future(asset_name, date_string, expire_month)
        asset_code = row['symbol'].strip()
        symbol = get_future_symbol(asset_code, future_month_name, future_year_name)
        name = f"{row['name']} ({future_month_name})"
    else:
        name = row['name'].strip()
        symbol = row['symbol']
    return name, symbol


def load_tsv_data(data_fp):
    df = pd.read_csv(data_fp, encoding='utf-16', sep='\t')
    df['date'] = pd.to_datetime(df['date']).dt.strftime("%Y-%m-%d")
    return df

def load_source_ref_data():
    table_data_source_ref_path = os.path.join(data_dir, "financial_instrument_reference.json")
    with open(table_data_source_ref_path, 'r') as f:
        table_data_source_ref_list = json.load(f)
    ref_df = pd.DataFrame(table_data_source_ref_list)
    return ref_df


def select_historical_data_by_period(delta_value, unit):
    if unit == 'months':
        offset_period = pd.DateOffset(months=delta_value)
    elif unit == 'weeks':
        offset_period = pd.DateOffset(weeks=delta_value)
    elif unit == 'days':
        offset_period = pd.DateOffset(days=(delta_value + 1))
    else:
        raise "Invalid offset period unit!!"

    def select_historical_data_fixed_month_delta(historical_data):
        selected_historical_data = historical_data[historical_data['Date'] > (historical_data['Date'].max() - offset_period)]
        return selected_historical_data
    return select_historical_data_fixed_month_delta


def setup_output_file(split, historical_data_offset_period):
    data_version_dir = f"{data_dir}/{historical_data_offset_period}"
    if not os.path.exists(data_version_dir):
        os.mkdir(data_version_dir)
    result_file_path = f"{data_version_dir}/{split}.json"
    return result_file_path



def update_json_with_new_items(json_path, new_items):
    if os.path.exists(json_path):
        with open(json_path, "r", encoding='utf-16') as f:
            content_list = json.load(f)
    else:
        content_list = []

    if isinstance(new_items, list):
        content_list = content_list + new_items
    else:
        content_list.append(new_items)

    with open(json_path, "w", encoding='utf-16') as f:
        json.dump(content_list, f, indent="\t", ensure_ascii=False)
    return content_list




def extract_table_data_single_entity(entity_ref_row, report_date, historical_data_process_func):
    entity_name, entity_symbol = get_entity_name_symbol(entity_ref_row)

    report_date_with_time = datetime.strptime(f"{report_date.year}-{report_date.month}-{report_date.day} 23:59:59", "%Y-%m-%d %H:%M:%S")
    mask = (table_df['Date'] <= report_date_with_time) & (table_df['Symbol'] == entity_symbol)
    df_table_single_entity_selected_period = table_df[mask]
    df_table_single_entity_selected_period = historical_data_process_func(df_table_single_entity_selected_period)
    df_table_single_entity_selected_period = df_table_single_entity_selected_period.sort_values(by='Date', ascending=True)

    return df_table_single_entity_selected_period


def make_table_output_dir(market):
    market_output_dir = os.path.join(table_data_output_dir, market)
    if not os.path.isdir(market_output_dir): os.mkdir(market_output_dir)
    return market_output_dir


def extract_table_data_of_required_period():
    ref_df = load_source_ref_data()
    historical_data_process_func = select_historical_data_by_period(
        int(re.search('\d+', historical_data_offset_period)[0]),
        unit=re.search('[a-z]+', historical_data_offset_period)[0])

    market_list = report_df['source'].unique().tolist()

    for _, report_row in tqdm(report_df.iterrows()):
        market_output_dir = make_table_output_dir(report_row['market'])
        selected_ref_df = ref_df[ref_df['market'] == report_row['market']]
        report_date = datetime.strptime(report_row['date'], '%Y-%m-%d')

        df_list = []
        for _, entity_ref_row in selected_ref_df.iterrows():
            df_table_single_entity = extract_table_data_single_entity(entity_ref_row, report_date, historical_data_process_func)
            df_list.append(df_table_single_entity)
        df_table_single_report = pd.concat(df_list)

        date_str = pd.to_datetime(report_date).strftime("%Y-%m-%d")
        table_output_path = os.path.join(market_output_dir, f"{date_str}.csv")
        df_table_single_report.to_csv(table_output_path, index=False)


def get_table_data_string_for_report(market, date_str):
    table_path = os.path.join(table_data_output_dir, market, date_str)
    df_table = pd.read_csv(table_path)
    return df_table.to_string(index=False)


def format_input_table(date_str, table_str):
    return f"Generate report for date {date_str}\nTable:\n{table_str}\nReport:"


def get_prompt_for_market(market, df_example, include_example_setting):
    data2text_task_instruction_market = data2text_task_instruction.strip().replace("[market]", market)

    single_str_prompt = data2text_task_instruction_market
    chat_prompt = [{"role": "system", "content": data2text_task_instruction_market}]
    for i, example_row in enumerate(df_example):
        date_str = pd.to_datetime(example_row['Date']).strftime("%Y-%m-%d")

        if include_example_setting == 'few-shot':
            df_example_table_string = get_table_data_string_for_report(market, date_str)
            df_example_table_string_formated = format_input_table(date_str, df_example_table_string)
            single_line_prompt += f"\nExample {i + 1}:\n{df_example_table_string_formated}\n{example_row['passage']}\n"
            chat_prompt += [
                {"role": "user", "content": df_example_table_string_formated},
                {"role": "assistant", "content": example_row['passage']}
            ]
        else:
            single_line_prompt += f"\nExample report for date {example_row['Date']}\n{example_row['passage']}\n"
            chat_prompt[0]['content'] += single_str_prompt

    return single_line_prompt, chat_prompt


def get_prompts_for_all_market():
    data2text_generation_task_instruction_path = "data2text_generation_task_instruction.txt"
    with open(data2text_generation_task_instruction_path, 'r', encoding='utf-8') as f:
        data2text_task_instruction = f.read()

    if include_example_as in ['report_sample', 'few_shot']:
        data2text_generation_example_path = os.path.join(data_dir, "generation_2_example_merged_2.tsv")
        df_examples = load_tsv_data(data2text_generation_example_path)
    else:
        df_examples = pd.DataFrame()
        print('Invalid example setting. Skipping example...')

    market_specific_prompt_dict = {}
    for _, report_row in df_report.iterrows():
        if (report_row['market'], report_row['source']) not in market_specific_prompt_dict.keys():
            df_examples_market = df_examples[
                (df_examples['market'] == report_row['market']) & (df_examples['source'] == report_row['source'])]
            single_line_prompt, chat_prompt = get_prompt_for_market(row['market'], df_examples_market)
            market_specific_prompt_dict[(report_row['market'], report_row['source'])] = {'single_line_prompt': single_line_prompt,
                                                                                         'chat_prompt': chat_prompt}
        else:
            continue
    return market_specific_prompt_dict


def setup_output_file(split, historical_data_offset_period):
    data_version_dir = f"{dataset_with_prompt_output_dir}/{historical_data_offset_period}"
    if not os.path.exists(data_version_dir):
        os.mkdir(data_version_dir)
    result_file_path = f"{data_version_dir}/{split}.json"
    return result_file_path


def prepare_dataset(include_example_as='report_sample'):
    market_specific_prompt_dict = get_prompts_for_all_market()
    split_ref_path = os.path.join(data_dir, "split_ref.csv")
    df_split_ref = pd.read_csv(split_ref_path)

    for data_split in ['test', 'validate', 'train']:
        processed_data_path = setup_output_file(data_split, historical_data_offset_period)

        df_split_ref_selected = df_split_ref[df_split_ref['split'] == data_split]
        df_report_split = df_split_ref_selected.merge(table_df, on=['source', 'market', 'date'])

        for _, report_row in tqdm(df_report_split.iterrows()):
            report_generation_prompts = market_specific_prompt_dict[(report_row['market'], report_row['source'])]

            cur_date_str = pd.to_datetime(example_row['Date']).strftime("%Y-%m-%d")
            cur_table_data_str = get_report_table_data(report_row['market'], date_str)

            example_i = {'source': report_row['source'],
                         'market': report_row['market'],
                         'date': cur_date_str,
                         'instruction': report_generation_prompts['single_line_prompt'],
                         'table_data': cur_table_data_str,
                         'report': report_row['passage']
                         }
            example_i.update(report_generation_prompts)
            update_json_with_new_items(processed_data_json_path, example_i)


def format_llm_input(example) -> dict:
    context = f"Instruction: {example['instruction']}\n"
    if example.get("input"):
        context += format_input_table(example['date'], example['input'])
    # context = tokenizer.decode(tokenizer.encode(context, truncation=True, max_length=context_length)[1:])
    # context = '\n'.join(context.split('\n')[:-1])
    target = example["output"]
    return {"context": context, "target": target}


def prepare_single_tokenized_dataset(tokenizer, config, historical_data_offset_period, output_data_dir, data_split, max_seq_length, max_output_length, skip_overlength):
    input_data_path = os.path.join(output_data_dir, historical_data_offset_period, f"{data_split}.json")
    output_data_path = os.path.join(tokenized_dataset_with_prompt_output_dir, data_split)

    with open(input_data_path, 'r', encoding='utf-16') as f:
        data_list = json.load(f)

    df_data = pd.DataFrame(data_list)
    df_data['id'] = df_data[['source', 'market', 'date']].agg('-'.join, axis=1)

    df_data = df_data[['id', "table_data", "report", "instruction"]]
    df_data.columns = ["id", "input", "output", "instruction"]
    df_data["input"] = df_data["input"].str.replace("'", "\\'").str.replace('"', '\\"')
    df_data[['context', 'target']] = df_data.apply(lambda x: format_llm_input(x), axis=1, result_type="expand")

    df_data['Date'] = df_data['id'].apply(lambda x: re.search(r'-([\d\-]+)', x).group(1))
    df_data = df_data.sort_values('Date')

    dataset = datasets.Dataset.from_generator(
        lambda: read_dataset(df_data, tokenizer, config, max_seq_length-max_output_length, skip_overlength)
    )
    dataset.save_to_disk(output_data_path)


def tokenize_dataset(model_name, max_length, skip_overlength, max_output_length):
    model_name_str = model_name.split('/')[-1]
    tokenized_dir = os.path.join(tokenized_dataset_with_prompt_output_dir, model_name_str)
    if not os.path.exists(tokenized_dir):
        os.mkdir(tokenized_dir)

    if 'llama' in model_name:
        tokenizer = LlamaTokenizerFast.from_pretrained(
            model_name, trust_remote_code=True)
        config = LlamaConfig.from_pretrained(
            model_name, trust_remote_code=True, device_map='auto')
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True)
        config = AutoConfig.from_pretrained(
            model_name, trust_remote_code=True, device_map='auto')

    output_data_version_dir = f"{tokenized_dir}/{historical_data_offset_period}"
    if not os.path.exists(output_data_version_dir):
        os.mkdir(output_data_version_dir)
    for data_split in ['test', 'validate', 'train']:
        prepare_single_tokenized_dataset(tokenizer, config,
                                         historical_data_offset_period, output_data_version_dir, data_split,
                                         max_length, max_output_length=max_output_length,
                                         skip_overlength=skip_overlength)


if __name__ == '__main__':
    historical_data_offset_period = '3months'
    data_dir = "data"

    report_path = os.path.join(data_dir, 'reports', "reports.tsv")
    report_df = load_tsv_data(report_path)
    all_table_data_path = os.path.join(data_dir, "all_table_data.csv")
    table_df = pd.read_csv(all_table_data_path)
    table_df['Date'] = pd.to_datetime(table_df['Date'], format='%Y-%m-%d')

    table_data_output_dir = os.path.join(data_dir, "table_data")
    dataset_with_prompt_output_dir = os.path.join(data_dir, "processed")
    tokenized_dataset_with_prompt_output_dir = os.path.join(data_dir, "tokenized")

    for output_dir in [table_data_output_dir, dataset_with_prompt_output_dir]:
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

    source_refs = load_source_ref_data()
    extract_table_data_of_required_period()
    # prepare_dataset()

    # MAX_LENGTH = 4096
    # tokenize_dataset(model_name="daryl149/llama-2-7b-chat-hf", max_length=MAX_LENGTH, skip_overlength=True,
    #                           max_output_length=512)

