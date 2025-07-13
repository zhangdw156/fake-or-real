import argparse
import torch
import os
from trl import SFTConfig,SFTTrainer
from datasets import load_dataset
from peft import LoraConfig,PeftModel
import swanlab
from transformers import pipeline,AutoModelForCausalLM,AutoTokenizer
import pandas as pd


def parse_args():
    parser=argparse.ArgumentParser(description='sft训练')

    parser.add_argument('--model_path', type=str,help='微调模型的路径')
    parser.add_argument('--lora_path', type=str,help='lora参数的路径')
    parser.add_argument('--data_path', type=str,help='训练数据的路径')
    parser.add_argument('--job_name', type=str,help='任务名称')
    parser.add_argument('--epochs', type=int,default=10,help='轮数')
    parser.add_argument('--batch_size', type=int,default=1,help='批次')
    parser.add_argument('--lora', 
                        action='store_true',
                        dest='lora',
                        help='启用LoRA')
    parser.set_defaults(lora=False) 
    
    args = parser.parse_args()

    return args

def main():
    args=parse_args()
    model_path=args.model_path
    data_path=args.data_path
    job_name=args.job_name
    epochs=args.epochs
    batch_size=args.batch_size
    
    model_name=model_path.split('/')[-1]
    

    ## 纪录训练过程
    swanlab.config.update({
        "model": model_name
    })

    ## 加载训练数据
    train_ds=load_dataset('json',data_files=data_path,split='train')
    ## 加载模型和分词器
    tokenizer=AutoTokenizer.from_pretrained(model_path)
    model=AutoModelForCausalLM.from_pretrained(model_path)

    if args.lora:
        model=PeftModel.from_pretrained(model,args.lora_path)
        model=model.merge_and_unload()
        print(f"已经合并lora：{args.lora_path}")

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
        per_device_train_batch_size = batch_size,
        gradient_accumulation_steps = 1, # Use GA to mimic batch size!
        warmup_steps = 5,
        num_train_epochs = epochs, 
        learning_rate = 5e-5, 
        report_to = "swanlab", 
        run_name=f"{model_name}/{job_name}",
        output_dir=f"/data/finetuning/tof/{model_name}/{job_name}",
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
    trainer.save_model(f'model/{model_name}/{job_name}/')
    

if __name__=='__main__':
    main()


    