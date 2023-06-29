import pandas as pd
import json
import datetime,time
import shutil
import os
import re
import subprocess
from sklearn.metrics import classification_report
from pynvml import (nvmlInit, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex,
                    nvmlDeviceGetName, nvmlDeviceGetMemoryInfo, nvmlShutdown)
from configs.common_config import *

model_loaded = False
project_change = False
last_lora_name = ''
max_new_tokens = 1500
generation_config = dict(
            temperature=0.001,
            top_k=30,
            top_p=0.85,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.2,
            max_new_tokens=max_new_tokens
            )
def read_excel_file(file_path):
    df = pd.read_excel(file_path)
    return df

def save_to_excel(df, file_path):
    df.to_excel(file_path, index=False)

def process_data(training_data_path):
     # 读取 Excel 文件
    df = pd.read_excel(training_data_path)
    log = []
    log.append(f'开始处理数据')
    
    all_data = []
    # 遍历每一行数据
    for index, row in df.iterrows():
        instruction = row['系统指示']
        question = row['问题']
        answer = row['回答']
        
        # 创建字典并将数据添加到列表中
        data = {"instruction": instruction, "input": question, "output": answer}
        all_data.append(data)

    log = '\n'.join(log)  # 使用换行符拼接log内容
    return all_data, log


def get_available_gpu(threshold=20000):
    # Initialize NVML
    nvmlInit()

    # Get the number of GPU devices
    device_count = nvmlDeviceGetCount()

    # Find GPU devices with available memory greater than the threshold
    available_gpus = []
    for i in range(device_count):
        handle = nvmlDeviceGetHandleByIndex(i)
        info = nvmlDeviceGetMemoryInfo(handle)
        free_memory_mb = info.free / 1024 / 1024

        if free_memory_mb > threshold:
            available_gpus.append(i)

    # Shutdown NVML
    nvmlShutdown()

    return available_gpus

def pulse_train_model(model_name, lora_name,  training_data_path):
    now_str = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    print('now_str',now_str)
    all_data,log = process_data(training_data_path)
    log_file_path = f'data/logs/{now_str}.log'  # 定义log文件路径
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)  # 创建存储log的文件夹

    with open(log_file_path, 'w', encoding="utf-8") as f:
        f.write(log)  # 将log内容写入文件
    with open(f"data/{lora_name}_dataset.json", "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=4,  ensure_ascii=False)
    if not os.path.exists('finetune/pulse/data'):
        os.makedirs('finetune/pulse/data')
    if not os.path.exists('finetune/pulse/logs'):
        os.makedirs('finetune/pulse/logs')
    shutil.copyfile(f"data/{lora_name}_dataset.json", f"finetune/pulse/data/{lora_name}_dataset.json")
    
    available_gpus = get_available_gpu(threshold=20000)
    print('available_gpus[0]',available_gpus[0])
    content = f'''python convert_to_conv_data.py --orig_data data/{lora_name}_dataset.json --write_data  data/{lora_name}_dataset_conv.json --dataset_name {lora_name}

CUDA_VISIBLE_DEVICES={available_gpus[0]} torchrun --nproc_per_node 1 finetune.py --model_name_or_path {llm_model_dict[model_name]["local_model_path"]} --use_lora True --use_int8_training --lora_config configs/lora_config_bloom.json --train_file data/{lora_name}_dataset_conv.json --validation_file data/{lora_name}_dataset_conv.json --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 2 --num_train_epochs 2 --model_max_length 100 --save_strategy "steps" --save_total_limit 3 --learning_rate 3e-4 --weight_decay 0.00001 --warmup_ratio 0.05 --lr_scheduler_type "cosine" --logging_steps 10 --evaluation_strategy "steps" --seed 2048 --gradient_checkpointing True --cache_dir cache/{lora_name} --output_dir output/{lora_name}
    '''
    sh_file_name = f'finetune/pulse/train_{lora_name}.sh'

    with open(sh_file_name , 'w') as file:
        file.write(content)

    # 设置文件可执行权限
    os.chmod(sh_file_name , 0o755)
    now_str = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    print('now_str',now_str)
    subprocess.Popen(f"""cd finetune/pulse && . /home/pai/etc/profile.d/conda.sh && conda activate med_llm && nohup sh train_{lora_name}.sh > ./logs/train_{now_str}.log 2>&1 &""", shell=True)
    print('finish')
    # model.train(training_data_path)
    return f'{model_name} on training'

def stop_train_process():
    process = subprocess.Popen('ps -ef | grep finetune.py', shell=True, stdout=subprocess.PIPE)
    output, _ = process.communicate()
    process.kill()

    
    
    n = 0
    # 解析输出以获取进程ID
    print('output',output)
    try:
        lines = output.decode().split('\n')
        for line in lines:
            if 'finetune.py' in line:
                parts = line.split()
                pid = parts[1]
                # 杀死进程
                subprocess.call(['kill', '-9', pid])
                n+=1
    except Exception as e:
        print('error!!',e)

    return f'停止了{n//2}个进程'