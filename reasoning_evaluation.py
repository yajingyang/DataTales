import os
import time


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,3"
import openai

from tqdm import tqdm
import torch

# Load pre-trained model and tokenizer from Hugging Face
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, LogitsProcessor, LogitsProcessorList
from transformers import LogitsProcessorList, PrefixConstrainedLogitsProcessor

import re
from typing import List, Callable, Iterable
import json
import pandas as pd

from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, LlamaTokenizerFast, BitsAndBytesConfig   # 4.30.2

import math
from sklearn.metrics import accuracy_score
import stanza


openai.api_type = "azure"
openai.api_base = "https://oairtdsdevpacetest01.openai.azure.com/"
openai.api_version = "2023-03-15-preview"
openai.api_key = os.getenv("OPENAI_API_KEY")
def get_gpt_response(message):
    response = openai.ChatCompletion.create(
        engine=eval_model,
        messages=message,
        temperature=0.5,
        max_tokens=2000,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None)
    return response



class NumericLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, token_pos, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.token_pos = token_pos

        self.numeric_tokens = [i for token, i in self.tokenizer.get_vocab().items() if re.fullmatch(r"[0-9][0-9\.\,\%]*", token.strip())]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Mask all non-numeric tokens at the desired position
        if input_ids.size(1) == self.token_pos:
            for token_idx in range(scores.shape[1]):
                if token_idx not in self.numeric_tokens:
                    scores[0][token_idx] = -math.inf

        return scores


def update_json_with_new_items(json_path, new_items):
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            content_list = json.load(f)
    else:
        content_list = []

    if isinstance(new_items, list):
        content_list = content_list + new_items
    else:
        content_list.append(new_items)

    with open(json_path, "w") as f:
        json.dump(content_list, f, indent='\t')
    return content_list


def load_llama_tokenizer_model(base_model):
    tokenizer = LlamaTokenizerFast.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    # Quantization
    q_config = BitsAndBytesConfig(load_in_4bit=True,
                                  bnb_4bit_quant_type='nf4',
                                  bnb_4bit_use_double_quant=True,
                                  bnb_4bit_compute_dtype=torch.float16
                                  )

    model = LlamaForCausalLM.from_pretrained(
        base_model,
        quantization_config=q_config,
        load_in_8bit=True,
        trust_remote_code=True,
        device_map="auto"
    )
    return tokenizer, model


def post_processing(text):
    processed_text = ''
    for c in text:
        if not (c.isnumeric() or c in ['.',',','%']):
            break
        else:
            processed_text += c

    processed_text = processed_text[:-1] if processed_text.endswith('.') else processed_text
    return processed_text


def generate_next_tokens(input_context, masked_sentence):
    instruction = "Based on the given context, predict the next token which should be a numeric or percentage value."
    if 'gpt' in eval_model:
        llm_input = \
            [
                {
                    "role": "system",
                    "content": instruction
                },
                {
                    "role": "user",
                    "content": f"Context: {input_context}\nSentence: {masked_sentence}"
                }
            ]
        num_try = 0
        output_text = None
        while output_text is None and num_try < 5:
            try:
                r = get_gpt_response(llm_input)
                output_text = r['choices'][0]['message']['content']
                output_text = post_processing(output_text)
                break
            except Exception as e:
                print(e)
                num_try += 1
            time.sleep(30)
    else:
        llm_input = f"Instruction: {instruction}\nContext: {input_context}\nSentence: {masked_sentence}"
        # Encode input and identify the position of <extra_id_0>
        input_ids = tokenizer.encode(llm_input, return_tensors="pt")
        extra_id_pos = len(input_ids[0].tolist())

        # Initialize and pass the custom logits processor
        numeric_processor = NumericLogitsProcessor(tokenizer, extra_id_pos)
        output_ids = model.generate(input_ids, logits_processor=[numeric_processor], max_new_tokens=5, do_sample=True,
                                    num_beams=2)

        # Decode the output
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return llm_input, output_text


def get_substrings_before_numerics(s):
    doc = nlp(s)
    num_ners = ['PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL']
    ner_numeric_tokens = [ent.text for sent in doc.sentences for ent in sent.ents if any([num_ner in ent.type for num_ner in num_ners])]

    # Find all occurrences of numeric values and their indexes in the string
    numeric_occurrences = [(m.start(0), m.group(0)) for m in re.finditer(r'\b\d+(\.\d+)*(\%)?', s) if m.group(0) in ner_numeric_tokens]

    # Extract substrings before each numeric value
    substrings = [s[:idx] for idx, _ in numeric_occurrences]
    numeric_words = [s for _, s in numeric_occurrences]

    return substrings, numeric_words


def next_token_prediction_based_evaluation(input_path, output_path, text_column):
    print(f"Evaluating {input_path}")
    with open(input_path, 'r') as f:
        data_list = json.load(f)

    eval_result_list = []
    for data_i in tqdm(data_list):
        if 'acc_eval_result' in data_i.keys():
            data_i_processed_results = []
            for result_i in data_i['acc_eval_result']:
                output_text = result_i['numeric_word_predicted']
                if 'gpt' in eval_model:
                    out_text = output_text
                else:
                    try:
                        out_text = output_text.split(result_i['substring_before_numerics'])[1].split()[0].strip()
                        out_text = re.search(r"\d+(?:,\d+)*(?:\.\d+)?%?", out_text).group(0)
                    except:
                        out_text = ""
                result_i['numeric_word_predicted'] = out_text
                result_i['acc_score'] = 1 if out_text == result_i['numeric_word'] else 0
                eval_result_list.append(result_i)
                data_i_processed_results.append(result_i)
            data_i['acc_eval_result'] = data_i_processed_results
        else:
            substrings_before_numerics_list, numeric_word_list = get_substrings_before_numerics(data_i[text_column])
            data_i['acc_eval_result'] = []
            for substring_before_numerics, numeric_word in zip(substrings_before_numerics_list, numeric_word_list):
                result_i = {}
                input_text, output_text = generate_next_tokens(data_i['input'], substring_before_numerics)
                result_i['substring_before_numerics'] = substring_before_numerics
                result_i['acc_eval_input'] = input_text

                if 'gpt' in eval_model:
                    out_text = output_text
                else:
                    try:
                        out_text = output_text.split(substring_before_numerics)[1].split()[0].strip()
                        out_text = re.search(r"\d+(?:,\d+)*(?:\.\d+)?%?", out_text).group(0)
                    except:
                        out_text = ""
                result_i['numeric_word'] = numeric_word
                result_i['numeric_word_predicted'] = out_text
                result_i['acc_score'] = 1 if out_text == numeric_word else 0
                result_i['id'] = data_i['id']
                eval_result_list.append(result_i)

                data_i['acc_eval_result'].append(result_i)
        update_json_with_new_items(output_path, data_i)

    df_eval_result = pd.DataFrame(eval_result_list)
    return df_eval_result


def evaluate_generation_results(dataset, evaluation_result_path, model_name_str, historical_postfix, is_finetuned, context_length, text_column, skip_overlength_market=False):
    # if text_column == 'target':
    #     metric_result = {
    #         'model': 'Human Expert',
    #         'historical_data': 'All',
    #         'is_finetuned': 'N.A.',
    #         'num_examples': len(dataset),
    #         'context_length': 'N.A.',
    #         'skip_overlength_market': skip_overlength_market,
    #         'acc_score': dataset['acc_score'].mean(),
    #         'acc_count': dataset['acc_count'].mean()
    #     }
    # else:
    metric_result = {
        'model': model_name_str,
        'historical_data': historical_postfix,
        'is_finetuned': is_finetuned,
        'num_examples': len(dataset),
        'context_length': context_length,
        'skip_overlength_market': skip_overlength_market,
        'acc_score': dataset['acc_score'].mean(),
        'acc_count': dataset['acc_count'].mean()
    }
    update_json_with_new_items(evaluation_result_path, metric_result)
    return metric_result

def combine_acc_eval_by_instance(df_eval):
    df_score = df_eval.groupby('id')['acc_score'].mean()
    df_score = df_score.reset_index()

    df_count = df_eval.groupby('id').apply(len)
    df_count = df_count.reset_index()
    df_count.columns = ['id', 'acc_count']

    df_eval_agg = df_score.merge(df_count, on='id')
    return df_eval_agg


if __name__ == '__main__':
    eval_model = 'llama2-chat-7b'
    skip_predict = True
    if 'llama' in eval_model and not skip_predict:
        tokenizer, model = load_llama_tokenizer_model(eval_model)
    else:
        tokenizer, model = None, None

    # input_sequences = [
    #     "The closing price of Lean Hog Index is 215.80 yesterday and 215.86 today.\nThe Lean Hog Index traded <extra_id_0> higher to 215.86.",
    #     "dollar index trend: day 1: down, day 2: up, day 3: up, day 4: up\nThe dollar index trend up for <extra_id_0> days."
    # ]
    #
    # input_context_list = [
    #     "The closing price of Lean Hog Index is 215.80 yesterday and 215.86 today.",
    #     "dollar index trend: day 1: down, day 2: up, day 3: up, day 4: up"
    # ]
    #
    # input_masked_sentence_list = [
    #     "The Lean Hog Index traded ",
    #     "The dollar index trend up for "
    # ]

    # input_text = "The closing price of Lean Hog Index is 215.80 yesterday and 215.86 today.\nThe Lean Hog Index traded <extra_id_0> higher to 215.86."
    nlp = stanza.Pipeline('en', processors='tokenize,ner')

    context_length = 4096
    inference_model_list = ["gpt-35-turbo"]
    is_finetuned_settings = [False]
    historical_postfix_settings = ["0days"]

    text_column = 'target'

    # file_list = ['acc_eval_result_llama-2-7b-chat-hf_0days_0days_finetune_True_context.json']
    file_list = [f for f in os.listdir('evaluation_results/evaluation_2')]
    # file_to_evaluate = "evaluation_results/acc_eval_human_report_result_Llama-2-7b-chat-hf_0days_finetune_False.json"

    # for inference_model in inference_model_list:
    #     for historical_postfix in historical_postfix_settings:
    #         for is_finetuned in is_finetuned_settings: # True,
    #             # generation_result_path = f"../generation_results/results_{inference_model}_{historical_postfix}_finetune_{is_finetuned}_context_{context_length}.json"
    #             generation_result_path = file_to_evaluate
    #             output_path = f'../evaluation/acc_eval_human_report_{eval_model}_{historical_postfix}_finetune_{is_finetuned}.json'
    #             eval_result_path = "../generation_results/acc_evaluation_scores.json"
    #
    #             df_eval_output = next_token_prediction_based_evaluation(generation_result_path, output_path, text_column=text_column)
    #             df_eval_output_by_instance = combine_acc_eval_by_instance(df_eval_output)
    #             evaluate_generation_results(df_eval_output_by_instance, eval_result_path, eval_model, historical_postfix,
    #                                         is_finetuned, context_length, text_column)

    for fname in file_list:
        eval_model = 'llama2-7b' if '7b' in fname else 'llama2-13b'
        historical_postfix = '0days' if '0days' in fname else '1weeks'
        is_finetuned = True if 'True' in fname else False
        # generation_result_path = f"../generation_results/results_{inference_model}_{historical_postfix}_finetune_{is_finetuned}_context_{context_length}.json"
        generation_result_path = os.path.join('evaluation_results/evaluation_2', fname)
        output_path = f'../evaluation/acc_eval_post_processed_human_report_{eval_model}_{historical_postfix}_finetune_{is_finetuned}.json'
        eval_result_path = "../generation_results/acc_evaluation_scores.json"

        df_eval_output = next_token_prediction_based_evaluation(generation_result_path, output_path, text_column=text_column)
        df_eval_output_by_instance = combine_acc_eval_by_instance(df_eval_output)
        evaluate_generation_results(df_eval_output_by_instance, eval_result_path, eval_model, historical_postfix,
                                    is_finetuned, context_length, text_column)
