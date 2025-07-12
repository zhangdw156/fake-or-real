import os
import argparse
from transformers import pipeline,AutoModelForCausalLM,AutoTokenizer
from peft import PeftModel
import logging
import pandas as pd
import numpy as np

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
    
    args = parser.parse_args()

    return args

def main():
    args=parse_args()
    logger=prepare_logger()
    model_path=args.model_path
    lora_path=args.lora_path
    model_name=model_path.split('/')[-1]
    
    logger.info('***********************************************')
    logger.info(f'微调的基础模型是{model_name}')
    logger.info(f'lora所在的位置是{lora_path}')
    
    tokenizer=AutoTokenizer.from_pretrained(model_path)
    model=AutoModelForCausalLM.from_pretrained(model_path)

    model=PeftModel.from_pretrained(model,lora_path)
    model=model.merge_and_unload()

    generator = pipeline(
        task="text-generation",  # 指定任务类型
        model=model,             # 传入预加载的模型
        max_new_tokens=4,         # 其他参数
        tokenizer=tokenizer,
    )

    test_data=pd.read_csv('data/test_processed.csv')
    test_data['text']=test_data.apply(
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

    outputs=generator(test_data['text'].tolist())

    outputs=[row[0]['generated_text'][len(test_data.loc[idx,'text']):] for idx,row in enumerate(outputs)]

    test_data['real_text_id']=outputs

    ## 如果不是1或者2，则随机设置为1或者2
    test_data['real_text_id']=test_data['real_text_id'].astype(str)

    mask = ~test_data['real_text_id'].isin(['1', '2'])  

    test_data.loc[mask, 'real_text_id'] = np.random.choice(['1', '2'], size=mask.sum())  

    test_data['real_text_id']=test_data['real_text_id'].astype(int)
    
    test_data=test_data[['id','real_text_id']]
    
    test_data.to_csv(f'data/submission_{model_name}_sft.csv',header=True,index=False)


if __name__=='__main__':
    main()







    