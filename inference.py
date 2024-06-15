import warnings

warnings.filterwarnings("ignore")
import os

from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, LlamaTokenizerFast, BitsAndBytesConfig   # 4.30.2
from peft import PeftModel  # 0.4.0

import evaluate
from datasets import load_dataset
from tqdm import tqdm
import datasets
import torch
import json
import pandas as pd
import openai
import time

bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
# meteor = evaluate.load("meteor")
metric_dict = {"bleu": bleu, "rouge": rouge}

openai.api_type = "azure"
openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_version = "2023-03-15-preview"


def format_example(example, context_length, model):
    if 'gpt' in model:
        context = [{"role": "system", "content": example['instruction']}]
        if example.get("input"):
            context.append({"role": "user", "content": f"{example['input'].replace(',', '')}\n"})
    else:
        context = f"Instruction: {example['instruction']}\n"
        if example.get("input"):
            context += f"Input: {example['input'].replace(',', '')}\n"
        context = tokenizer.decode(tokenizer.encode(context, truncation=True, max_length=context_length-10)[1:])
        context = '\n'.join(context.split('\n')[:-1])
        context += "\n\nAnswer: "
    target = example["output"]
    return {"context": context, "target": target}


def get_gpt_response(message, engine):
    response = openai.ChatCompletion.create(
        engine=engine,
        messages=message,
        temperature=0.3,
        # max_tokens=300,
        # top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None)
    return response

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


def load_fingen_dataset(input_data_path, context_length, max_output_length):
    with open(input_data_path, 'r', encoding='utf-16') as f:
        data_list = json.load(f)

    df_data = pd.DataFrame(data_list)
    df_data['id'] = df_data[['source', 'market', 'date']].agg('-'.join, axis=1)

    df_data = df_data[['id', "source_data", "human_report", "instruction"]]
    df_data.columns = ["id", "input", "output", "instruction"]
    df_data["input"] = df_data["input"].str.replace("'", "\\'").str.replace('"', '\\"')
    df_data[['context', 'target']] = df_data.apply(lambda x: format_example(x, context_length-max_output_length, model_name), axis=1, result_type="expand")
    return df_data


def evaluate_generation_results(dataset, evaluation_result_path, model_name_str, historical_postfix, is_finetuned, context_length, skip_overlength_market):
    metric_result = {
        'model': model_name_str,
        'historical_data': historical_postfix,
        'is_finetuned': is_finetuned,
        'num_examples': len(dataset),
        'context_length': context_length,
        'skip_overlength_market': skip_overlength_market
    }

    if skip_overlength_market == True:
        overlength_market = ['equity market', 'bitcoin']
        dataset['market'] = dataset['id'].str.split('-')[1]
        dataset = dataset[~dataset['market'].isin(overlength_market)]

    print(f"Evaluating data size: {len(dataset['output'])}, {len(dataset['target'])}")
    for metric_name, metric_evaluator in metric_dict.items():
        metric_result_i = metric_evaluator.compute(predictions=dataset["output"],
                                                   references=dataset["target"])
        metric_result[metric_name] = metric_result_i
        print(metric_result_i)

    update_json_with_new_items(evaluation_result_path, metric_result)
    return metric_result


def get_differing_parts(str1, str2):
    # Find the index where the strings start to differ
    for i, (char1, char2) in enumerate(zip(str1.replace('<s>', '').strip(), str2.replace('<s>', '').strip())):
        if char1 != char2:
            break
    else:
        i = min(len(str1), len(str2))

    # Return the differing parts
    return str1[i:], str2[i:]


def generate_results_for_non_gpt_model(model, tokenizer, dataset, output_path, batch_size, context_length):
    total_steps = dataset.shape[0] // batch_size if dataset.shape[0] % batch_size == 0 else dataset.shape[0] // batch_size + 1
    context = dataset['context'].tolist()
    print(f"Total len: {len(context)}. Batchsize: {batch_size}. Total steps: {total_steps}")

    out_text_list = []
    for i in  tqdm(range(total_steps)):
        tmp_context = context[i * batch_size:(i + 1) * batch_size]
        tokens = tokenizer(tmp_context, return_tensors='pt', padding=True, max_length=context_length)
        # tokens.pop('token_type_ids')
        for k in tokens.keys():
            tokens[k] = tokens[k].cuda()

        # res = model.generate(**tokens, max_length=context_length, do_sample=True, temperature=0.2)
        res = model.generate(**tokens, max_length=context_length)
        res_sentences = [tokenizer.decode(i) for i in res]
        out_text = [o.split("Answer:")[1].split(tokenizer.eos_token)[0].strip() if len(o.split("Answer:")) > 1 else o for o in res_sentences ]
        out_text_list += out_text
        torch.cuda.empty_cache()
        df_cur_dataset_with_output = dataset[i * batch_size:(i + 1) * batch_size]
        df_cur_dataset_with_output['output'] = out_text
        cur_dataset_with_output = df_cur_dataset_with_output.to_dict('records')
        # print(cur_dataset_with_output[0])

        update_json_with_new_items(output_path, cur_dataset_with_output)
    return out_text_list


def generate_results_for_gpt_model(model, dataset, output_path):
    # context = dataset['context'].tolist()
    print(f"Total steps: {len(dataset)}")

    out_text_list = []
    for row in tqdm(dataset.to_dict(orient="records")):
        out_text = None
        num_attempt = 0
        while out_text is None and num_attempt < 5:  # to retry after API call failure
            try:
                response = get_gpt_response(row['context'], engine=model)['choices'][0]
                out_text = response['message']['content']
            except Exception as error:
                # handle the exception
                print("An exception occurred:", error)
                num_attempt += 1
                time.sleep(5)
                continue
        out_text_list.append(out_text)
        row['output'] = out_text

        update_json_with_new_items(output_path, row)
    return out_text_list


def test_fingen(model, tokenizer, historical_postfix, model_name, is_finetuned, batch_size=8, context_length=2048, max_output_length=200):
    input_data_path = f"../dataset/{historical_postfix}/test.json"
    model_name_str = model_name.lower().replace('.', '_').replace(' ', '_')
    output_path = f"../generation_results/results_{model_name_str}_{historical_postfix}_finetune_{is_finetuned}_context_{context_length}.json"
    evaluation_result_path = f"../generation_results/evaluation_scores.json"

    dataset = load_fingen_dataset(input_data_path, context_length, max_output_length=max_output_length)
    # dataset[["context", "target"]] = dataset.apply(format_example, axis=1, result_type="expand")
    print(f"\n\nPrompt example:\n{dataset['context'].iloc[0]}\n\n")

    print(f"\nGenerating result for {model_name_str}...")
    print(f'Saving generation results to {output_path}')
    if 'gpt' in model_name_str:
        out_text_list = generate_results_for_gpt_model(model_name_str, dataset, output_path)
    else:
        out_text_list = generate_results_for_non_gpt_model(model, tokenizer, dataset, output_path, batch_size, context_length)
    dataset["output"] = out_text_list

    print(f"\nEvaluating the generation results for {model_name_str}...")
    evaluate_generation_results(dataset, evaluation_result_path, model_name_str, historical_postfix, is_finetuned, context_length, skip_overlength_market=False)
    evaluate_generation_results(dataset, evaluation_result_path, model_name_str, historical_postfix, is_finetuned, context_length, skip_overlength_market=True)

    return dataset


def load_tokenizer_and_model(base_model_name, peft_model_name=None):
    if 'llama' in base_model_name:
        # Load tokenizer & model
        tokenizer = LlamaTokenizerFast.from_pretrained(base_model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
         # Quantization
        q_config = BitsAndBytesConfig(load_in_4bit=True,
                                      bnb_4bit_quant_type='nf4',
                                      bnb_4bit_use_double_quant=True,
                                      bnb_4bit_compute_dtype=torch.float16
                                      )

        model = LlamaForCausalLM.from_pretrained(
                base_model_name,
                # quantization_config=q_config,
                load_in_8bit = True,
                trust_remote_code=True,
                device_map="auto"
            )
        # model = torch.compile(model)  # Please comment this line if your platform does not support torch.compile

    else:
        # Load tokenizer & model
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            base_model_name,
            # quantization_config=q_config,
            load_in_8bit=True,
            trust_remote_code=True,
            device_map="auto"
        )

    if isinstance(peft_model_name, list):
        for single_peft_model_name in peft_model_name:
            model = PeftModel.from_pretrained(model, single_peft_model_name)
    elif isinstance(peft_model_name, str):
        model = PeftModel.from_pretrained(model, peft_model_name)

    model = model.eval()
    print(model)

    return model, tokenizer


def load_results_and_evaluate(model_name_str, historical_postfix, is_finetuned, context_length, skip_overlength_market):
    evaluation_result_path = f"../generation_results/evaluation_scores.json"
    generation_result_path = f"../generation_results/results_{model_name_str}_{historical_postfix}_finetune_{is_finetuned}_context_{context_length}.json"
    with open(generation_result_path, 'r') as f:
        data_list = json.load(f)
    df_dataset = pd.DataFrame(data_list)
    print(len(df_dataset))

    evaluate_generation_results(df_dataset, evaluation_result_path, model_name_str, historical_postfix, is_finetuned,
                                context_length=context_length, skip_overlength_market=skip_overlength_market)


if __name__ == '__main__':
    # base_model = "daryl149/llama-2-7b-chat-hf"
    base_model = 'gpt-35-turbo'
    base_model_name = base_model.split('/')[-1]
    finetune_settings = [False]
    historical_data_settings = ['0days', '1weeks']

    for is_finetuned in finetune_settings:
        for historical_data_postfix in historical_data_settings:
            if is_finetuned:
                finetune_model = f"../finetuned_model/{base_model_name}_{historical_data_postfix}/"
            else:
                finetune_model = None

            model_name = base_model_name
            peft_model = finetune_model

            if 'gpt' in model_name:
                print(f"Loading model from {model_name}...")
                model = None
                tokenizer = None
            else:
                print(f"Loading model from {peft_model}...")

                model, tokenizer = load_tokenizer_and_model(base_model, peft_model)
            print(f"\nEvaluating {model_name}...")

            try:
                test_fingen(model, tokenizer, historical_data_postfix, model_name=model_name, is_finetuned=is_finetuned, batch_size=4,
                            context_length=4096, max_output_length=512)
            except Exception as e:
                print(e)
