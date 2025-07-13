import os
import argparse
from transformers import pipeline,AutoModelForSequenceClassification,AutoTokenizer
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
    parser.add_argument('--data_path', type=str,help='验证数据的路径')
    
    args = parser.parse_args()

    return args

def main():
    args=parse_args()
    logger=prepare_logger()
    model_path=args.model_path
    lora_path=args.lora_path
    data_path=args.data_path
    model_name=model_path.split('/')[-1]
    
    logger.info('***********************************************')
    logger.info(f'微调的基础模型是{model_name}')
    logger.info(f'lora所在的位置是{lora_path}')
    
    tokenizer=AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, 
        num_labels=2,  # 二分类问题
    )
    model=PeftModel.from_pretrained(model,lora_path)
    model=model.merge_and_unload()

    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        return_all_scores=True,  # 返回所有类别的分数（0和1的概率）
    )

    train_df=pd.read_json(data_path,orient='records',lines=True)
    train_df['text'] = train_df['text'].replace('', ' ')  # 替换空字符串
    train_df_1=train_df[:95]
    train_df_2=train_df[95:]

    outpus_1=classifier(train_df_1['text'].tolist())
    outpus_2=classifier(train_df_2['text'].tolist())

    predict_1 = [row[1]['score'] for row in outpus_1]
    predict_2 = [row[1]['score'] for row in outpus_2]

    train_csv=pd.read_csv('data/train.csv')

    train_csv['predict_1']=predict_1
    train_csv['predict_2']=predict_2

    train_csv['predict'] = np.where(train_csv['predict_1'] > train_csv['predict_2'], 1, 2)

    train_csv['real_text_id']=train_csv['real_text_id'].astype(int)
    accuracy = (train_csv['real_text_id'] == train_csv['predict']).mean()
    logger.info(f"在训练集上的准确率为: {accuracy:.5f}")
    logger.info('***********************************************')


if __name__=='__main__':
    main()







    