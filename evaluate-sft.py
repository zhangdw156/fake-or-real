import os
import argparse
from transformers import pipeline,AutoModelForCausalLM,AutoTokenizer
from peft import PeftModel
import logging
import pandas as pd
import time

def prepare_logger():
    # 创建日志器
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)  # 设置日志级别为 DEBUG
    
    # 创建文件处理器
    file_handler = logging.FileHandler('record.log', encoding='utf-8')
    file_handler.setLevel(logging.INFO)  # 文件中只记录 INFO 及以上级别
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)  # 控制台显示 DEBUG 及以上级别
    
    # 创建格式化器并添加到处理器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 将处理器添加到日志器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
    


def parse_args():
    parser=argparse.ArgumentParser(description='评估模型')

    parser.add_argument('--model_path', type=str,help='微调模型的路径')
    parser.add_argument('--lora_path', type=str,help='lora参数的路径')
    parser.add_argument('--data_path', type=str,help='验证数据的路径')
    parser.add_argument('--no-lora', 
                        action='store_false',
                        dest='lora',
                        help='禁用LoRA（默认启用）')
    parser.set_defaults(lora=True)  # 默认启用LoRA
    
    args = parser.parse_args()

    return args

def main():
    args=parse_args()
    logger=prepare_logger()
    model_path=args.model_path
    data_path=args.data_path
    model_name=model_path.split('/')[-1]
    
    logger.info('***********************************************')
    logger.info(f'微调的基础模型是{model_name}')
    logger.info(f'验证的数据是{data_path}')
    
    tokenizer=AutoTokenizer.from_pretrained(model_path)
    model=AutoModelForCausalLM.from_pretrained(model_path)
    if args.lora:
        lora_path=args.lora_path
        logger.info(f'lora所在的位置是{lora_path}')
        model=PeftModel.from_pretrained(model,lora_path)
        model=model.merge_and_unload()

    generator = pipeline(
        task="text-generation",  # 指定任务类型
        model=model,             # 传入预加载的模型
        max_new_tokens=4,         # 其他参数
        tokenizer=tokenizer,
    )

    train_data=pd.read_json(data_path,orient='records',lines=True)

    train_data['text']=train_data.apply(
        lambda row:
            tokenizer.apply_chat_template(
                [
                    {'role':'user','content':row['prompt']}
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            ),
        axis=1
    )
    begin_time=time.time()
    
    outputs=generator(train_data['text'].tolist())
    
    end_time=time.time()
    outputs=[row[0]['generated_text'][len(train_data.loc[idx,'text']):] for idx,row in enumerate(outputs)]

    train_data['predict']=outputs

    train_data['completion']=train_data['completion'].astype(str)
    accuracy = (train_data['predict'] == train_data['completion']).mean()
    logger.info(f"在训练集上的准确率为: {accuracy:.5f}")
    logger.info(f"处理时间为: {(end_time-begin_time):.2f}s")
    logger.info('***********************************************')


if __name__=='__main__':
    main()







    