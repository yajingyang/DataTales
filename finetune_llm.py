import os


from typing import List, Dict, Optional
import json
import datasets
from datasets import Dataset
import torch
from torch.utils.tensorboard import SummaryWriter
from transformers.integrations import TensorBoardCallback

from loguru import logger
from datasets import load_dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, LlamaTokenizerFast   # 4.30.2

from peft import (
    TaskType,
    LoraConfig,
    PeftModel,
    get_peft_model,
    set_peft_model_state_dict,
    prepare_model_for_kbit_training,
    prepare_model_for_int8_training,
)
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING


def load_tokenizer_and_model(base_model_name, peft_model_name=None):
    if 'llama' in base_model_name:
        # Load tokenizer & model
        tokenizer = LlamaTokenizerFast.from_pretrained(base_model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token

        model = LlamaForCausalLM.from_pretrained(
                base_model_name,
                # quantization_config=q_config,
                load_in_8bit = True,
                trust_remote_code=True,
                device_map="auto"
            )

    else:
        # Load tokenizer & model
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            base_model_name,
            # quantization_config=q_config,
            load_in_8bit=True,
            trust_remote_code=True,
            device_map='auto'
        )

    if isinstance(peft_model_name, list):
        for single_peft_model_name in peft_model_name:
            model = PeftModel.from_pretrained(model, single_peft_model_name, is_trainable=True)
    elif isinstance(peft_model_name, str):
        model = PeftModel.from_pretrained(model, peft_model_name)

    model = prepare_model_for_int8_training(model, use_gradient_checkpointing=True)

    model = apply_lora(model, base_model_name)
    return model, tokenizer


def load_dataset_from_local_disk(model_str, historical_str):
    data_dir = f"../dataset/tokenized_{model_str}/{historical_str}"

    train_path = f"{data_dir}/train"
    train_dataset = datasets.load_from_disk(train_path)#.select(range(10))

    val_path = f"{data_dir}/validate"
    val_dataset = datasets.load_from_disk(val_path)#.select(range(10))
    return train_dataset, val_dataset


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

# LoRA
def apply_lora(model, base_model_name):
    if 'llama' in base_model_name:
        target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING['llama']
    else:
        target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING['chatglm']

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=target_modules,
        bias='none',
    )
    model = get_peft_model(model, lora_config)
    print_trainable_parameters(model)
    return model


def resume_training_from_checkpoint(resume_from_checkpoint):
    checkpoint_name = os.path.join(resume_from_checkpoint, 'pytorch_model.bin')
    if not os.path.exists(checkpoint_name):
        checkpoint_name = os.path.join(
            resume_from_checkpoint, 'adapter_model.bin'
        )
        resume_from_checkpoint = False
    if os.path.exists(checkpoint_name):
        logger.info(f'Restarting from {checkpoint_name}')
        adapters_weights = torch.load(checkpoint_name)
        set_peft_model_state_dict(model, adapters_weights)
    else:
        logger.info(f'Checkpoint {checkpoint_name} not found')


class ModifiedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        return model(
            input_ids=inputs["input_ids"],
            labels=inputs["labels"],
        ).loss

    def prediction_step(self, model: torch.nn.Module, inputs, prediction_loss_only: bool, ignore_keys = None):
        with torch.no_grad():
            res = model(
                input_ids=inputs["input_ids"].to(model.device),
                labels=inputs["labels"].to(model.device),
            ).loss
        return (res, None, None)

    def save_model(self, output_dir=None, _internal_call=False):
        from transformers.trainer import TRAINING_ARGS_NAME

        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        saved_params = {
            k: v.to("cpu") for k, v in self.model.named_parameters() if v.requires_grad
        }
        torch.save(saved_params, os.path.join(output_dir, "adapter_model.bin"))


def data_collator(features: list) -> dict:
    len_ids = [len(feature["input_ids"]) for feature in features]
    longest = max(len_ids)
    input_ids = []
    labels_list = []
    for ids_l, feature in sorted(zip(len_ids, features), key=lambda x: -x[0]):
        ids = feature["input_ids"]
        seq_len = feature["seq_len"]
        labels = (
            [tokenizer.pad_token_id] * (seq_len - 1) + ids[(seq_len - 1) :] + [tokenizer.pad_token_id] * (longest - ids_l)
        )
        ids = ids + [tokenizer.pad_token_id] * (longest - ids_l)
        _ids = torch.LongTensor(ids)
        labels_list.append(torch.LongTensor(labels))
        input_ids.append(_ids)
    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels_list)
    return {
        "input_ids": input_ids,
        "labels": labels,
    }


def finetune(resume_from_checkpoint=False, batch_size=2):
    training_args = TrainingArguments(
        output_dir=f'./finetuned_model/{model_name}_{historical_period_data_str}',  # saved model path
        logging_steps=100000,
        # max_steps=10000,
        num_train_epochs=5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        weight_decay=0.01,
        warmup_steps=1000,
        save_steps=100,
        fp16=True,
        # bf16=True,
        torch_compile=False,
        load_best_model_at_end=False,
        evaluation_strategy="steps",
        remove_unused_columns=False,
    )

    if resume_from_checkpoint:
        resume_training_from_checkpoint(resume_from_checkpoint)
        resume_from_checkpoint = False

    model.print_trainable_parameters()

    writer = SummaryWriter()
    trainer = ModifiedTrainer(
        model=model,
        args=training_args,             # Trainer args
        train_dataset=train_dataset, # Training set
        eval_dataset=val_dataset,   # Testing set
        data_collator=data_collator,    # Data Collator
        callbacks=[TensorBoardCallback(writer)],
    )
    trainer.train()
    writer.close()
    # save model
    model.save_pretrained(training_args.output_dir)


if __name__ == '__main__':
    base_model = "daryl149/llama-2-13b-chat-hf"
    base_model_name = base_model.split('/')[1]

    dataset_model_prefix = "llama-2-7b-chat-hf"
    peft_model = None
    model_name = base_model_name if peft_model is None else 'fingpt_v32_llama2'

    historical_period_data_str = "1weeks"
    train_dataset, val_dataset = load_dataset_from_local_disk(dataset_model_prefix, historical_period_data_str)
    print(len(train_dataset))
    model, tokenizer = load_tokenizer_and_model(base_model, peft_model)
    finetune()