import argparse
import torch
import os
from trl import SFTConfig,SFTTrainer
from datasets import load_dataset
from peft import LoraConfig
import swanlab
from transformers import pipeline,AutoModelForCausalLM,AutoTokenizer
import pandas as pd


def parse_args():
    parser=argparse.ArgumentParser(description='sft训练')

    parser.add_argument('--model_path', type=str,help='微调模型的路径')
    
    args = parser.parse_args()

    return args

def main():
    args=parse_args()
    model_path=args.model_path
    
    model_name=model_path.split('/')[-1]
    

    ## 纪录训练过程
    swanlab.config.update({
        "model": model_name
    })

    ## 加载训练数据
    train_ds=load_dataset('json',data_files='data/train_processed.json',split='train')
    ## 加载模型和分词器
    tokenizer=AutoTokenizer.from_pretrained(model_path)
    model=AutoModelForCausalLM.from_pretrained(model_path)

    ## lora参数
    lora_config=LoraConfig(
        r = 16,           # Choose any number > 0! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 32,  # Best to choose alpha = rank or rank*2
        lora_dropout = 0.1,
    )
    ## 准备训练
    sft_config = SFTConfig(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 1, # Use GA to mimic batch size!
        warmup_steps = 5,
        num_train_epochs = 10, 
        learning_rate = 5e-5, 
        report_to = "swanlab", 
        run_name=model_name,
        output_dir=f"/data/finetuning/tof/{model_name}/sft",
        completion_only_loss=True
    )
    trainer=SFTTrainer(
        model = model,
        processing_class = tokenizer,
        train_dataset = train_ds,
        args=sft_config,
        peft_config=lora_config,
    )

    ## 训练
    trainer.train()

    ## 保存
    trainer.save_model(f'model/{model_name}/sft/')
    

if __name__=='__main__':
    main()


    