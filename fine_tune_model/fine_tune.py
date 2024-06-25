import time
start_time = time.time()

import json
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, BitsAndBytesConfig,DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType
import torch
import dataset
import os
from dotenv import load_dotenv
# check if cuda is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n\n[ device: {device} ]\n\n")

# use your model path
load_dotenv()
model_path = os.getenv("origin_model")
output_model_path = os.getenv("fine_tuned_model")
training_loop = 25

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
) # quantization config
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  # 任務類型
    inference_mode=False,          # 訓練模式
    r=8,                           # Low-rank adaptation 參數
    lora_alpha=32,                 # LoRA 放大係數
    lora_dropout=0.1,              # LoRA dropout率
    bias="none",
) # 定義QLoRA配置
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=10,
    fp16=True  # 啟用FP16訓練
) # 訓練參數設置

# Tokenizer 加載
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=bnb_config
) # 加載4-bit量化模型

model = get_peft_model(model, lora_config) # 應用QLoRA到模型

# load dataset
train_dataset = Dataset.from_list(dataset.dataset['train'])
eval_dataset = Dataset.from_list(dataset.dataset['eval'])

def preprocess_function(examples):
    inputs = [f"<|begin_of_text|><|start_header_id|>時鐘<|end_header_id|>\n\n日期和時間: {st}<|eot_id|><|start_header_id|>{sn}<|end_header_id|>\n\n情緒: {se}\n話語: {ss}<|eot_id|><|start_header_id|>時鐘<|end_header_id|>\n\n日期和時間: {rt}<|eot_id|><|start_header_id|>{rn}<|end_header_id|>\n\n情緒: {re}\n話語: {rs}<|eot_id|><|end_of_text|>" for st, sn, se, ss, rt, rn, re, rs in zip(examples['說話的時間'], examples['說話者'], examples['說話者情緒'], examples['說話者話語'], examples['回應的時間'], examples['回應者'], examples['回應者情緒'], examples['回應者話語'],)]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding='max_length')
    return model_inputs

tokenized_train_datasets = train_dataset.map(preprocess_function, batched=True) # Preprocess the train dataset
tokenized_eval_datasets = eval_dataset.map(preprocess_function, batched=True) # Preprocess the eval dataset
# end
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_datasets,
    eval_dataset=tokenized_eval_datasets,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
) # 定義Trainer

end_time = time.time()
print(f"\n\n[ load take {int((end_time-start_time)*1000)}ms ]\n\n")
start_time = time.time()

for i in range(1 ,training_loop):
    trainer.train() # 微調模型
    print(f"\n\n[ current loop: {i} ]\n\n")

end_time = time.time()
print(f"\n\n[ train take {int((end_time-start_time))}s ]\n\n")

# 保存微調後的模型
model.save_pretrained(output_model_path)
tokenizer.save_pretrained(output_model_path)

# 評估模型（可選）
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")