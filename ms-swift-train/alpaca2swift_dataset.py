import json
import os
from modelscope.msdatasets import MsDataset
import random

def convert_to_jsonl(input_path, output_dir, max_train=5000, max_eval=500):
    # 确保输出路径的目录存在，如果不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建目录：{output_dir}")

    # 读取数据集
    ds = MsDataset.load(input_path, subset_name='default')

    # 打开输出文件
    train_path = os.path.join(output_dir, 'train.jsonl')
    eval_path = os.path.join(output_dir, 'eval.jsonl')

    # 初始化计数器
    train_count = 0
    eval_count = 0

    with open(train_path, 'w', encoding='utf-8') as f_train, open(eval_path, 'w', encoding='utf-8') as f_eval:
        for item in ds:
            # 提取数据内容
            instruction = item['instruction']+item['input']
            output = item['output']
            
            # 构造jsonl格式
            jsonl_content = {
                "messages": [
                    {"role": "system", "content": "你是一个专业的人工智能助手"},
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": output}
                ]
            }
            
            # 随机分配到训练集或评估集
            if random.random() < 0.9 and train_count < max_train:  # 90%的概率分配到训练集，直到达到最大训练数据量
                f_train.write(json.dumps(jsonl_content, ensure_ascii=False) + '\n')
                train_count += 1
            elif eval_count < max_eval:  # 10%的概率分配到评估集，直到达到最大评估数据量
                f_eval.write(json.dumps(jsonl_content, ensure_ascii=False) + '\n')
                eval_count += 1
            
            # 如果训练集和评估集都已达到最大数量，提前退出循环
            if train_count >= max_train and eval_count >= max_eval:
                break

    print(f"训练数据已成功保存到 {train_path}，共 {train_count} 条")
    print(f"评估数据已成功保存到 {eval_path}，共 {eval_count} 条")

# 使用示例
input_path = '/data/nvme0/zh_cot_110k_sft'
output_dir = './data'
convert_to_jsonl(input_path, output_dir)