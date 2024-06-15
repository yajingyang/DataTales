import os
import tiktoken
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, LlamaTokenizerFast, BitsAndBytesConfig   # 4.30.2
from sklearn.metrics import accuracy_score, f1_score
import evaluate
import pandas as pd


bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")

# meteor = evaluate.load("meteor")
metric_dict = {"bleu": bleu, "rouge": rouge}


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


def get_mean_length(dataset):
    get_length = lambda text:len(tokenizer.encode(text))
    output_length = dataset["output"].apply(get_length)
    return output_length.mean()


def evaluate_generation_results(dataset, evaluation_result_path, model_name_str, historical_postfix, is_finetuned, context_length):
    metric_result = {
        'model': model_name_str,
        'historical_data': historical_postfix,
        'is_finetuned': is_finetuned,
        'num_examples': len(dataset),
        'context_length': context_length,
    }

    print(f"Evaluating data size: {len(dataset['output'])}, {len(dataset['target'])}")
    for metric_name, metric_evaluator in metric_dict.items():
        metric_result_i = metric_evaluator.compute(predictions=dataset["output"].tolist(),
                                                   references=dataset["target"].tolist())
        if metric_name == 'mauve':
            metric_result[metric_name] = metric_result_i.mauve
        else:
            metric_result[metric_name] = metric_result_i
        print(metric_result_i)

    metric_result['token_length'] = get_mean_length(dataset)

    update_json_with_new_items(evaluation_result_path, metric_result)
    return metric_result


def load_results_and_evaluate(model_name_str, historical_postfix, is_finetuned, context_length):
    evaluation_result_path = f"../generation_results/evaluation_scores.json"
    generation_result_path = f"../generation_results/results_{model_name_str}_{historical_postfix}_finetune_{is_finetuned}_context_{context_length}.json"
    try:
        with open(generation_result_path, 'r') as f:
            data_list = json.load(f)
        df_dataset = pd.DataFrame(data_list)
        print(f'Evaluating {generation_result_path}')
        print(f"Number of results: {len(df_dataset)}")

        evaluate_generation_results(df_dataset, evaluation_result_path, model_name_str, historical_postfix, is_finetuned,
                                    context_length=context_length)
    except Exception as e:
        print(e)
