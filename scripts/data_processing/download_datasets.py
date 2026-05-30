"""
下载并准备LLM引导效应实验所需的基础数据集
"""
import os
import urllib.request
import zipfile
import json
from pathlib import Path
import time
import sys

# 设置Hugging Face缓存目录到scratch
os.environ['HF_HOME'] = '/scratch/yy5020/huggingface_cache'
os.environ['TRANSFORMERS_CACHE'] = '/scratch/yy5020/huggingface_cache/transformers'
os.environ['HF_DATASETS_CACHE'] = '/scratch/yy5020/huggingface_cache/datasets'

# 创建缓存目录
for cache_dir in ['/scratch/yy5020/huggingface_cache',
                 '/scratch/yy5020/huggingface_cache/transformers',
                 '/scratch/yy5020/huggingface_cache/datasets']:
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

# 确认缓存设置生效
print(f"Hugging Face缓存目录: {os.environ['HF_HOME']}")

# 导入datasets库(确保缓存目录已设置)
from datasets import load_dataset

# 创建数据目录
data_dir = Path("/scratch/yy5020/llm_priming_project/data")
raw_dir = data_dir / "raw"
processed_dir = data_dir / "processed"

for dir_path in [raw_dir / "cola", raw_dir / "blimp" / "data", processed_dir]:
    dir_path.mkdir(parents=True, exist_ok=True)

def download_file(url, target_path):
    """下载文件到指定路径"""
    print(f"Downloading {url} to {target_path}")
    urllib.request.urlretrieve(url, target_path)
    print(f"Download complete: {target_path}")

def extract_zip(zip_path, extract_to):
    """解压ZIP文件"""
    print(f"Extracting {zip_path} to {extract_to}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extraction complete: {extract_to}")

def download_cola():
    """下载CoLA数据集"""
    cola_url = "https://nyu-mll.github.io/CoLA/cola_public_1.1.zip"
    cola_zip = raw_dir / "cola" / "cola_public_1.1.zip"
    
    # 下载CoLA
    download_file(cola_url, cola_zip)
    
    # 解压CoLA
    extract_zip(cola_zip, raw_dir / "cola")
    
    print("CoLA dataset downloaded and extracted!")

def download_blimp():
    """下载BLiMP数据集（使用Hugging Face datasets）"""
    print("Downloading BLiMP dataset using Hugging Face datasets...")
    
    # 选择一些关键的语言现象 - 减少数量以节省空间
    phenomena = [
        "adjunct_island", 
        "determiner_noun_agreement_1",
        "regular_plural_subject_verb_agreement_1", 
        "wh_questions_object_gap"
    ]
    
    # 创建保存目录
    blimp_save_dir = raw_dir / "blimp" / "data"
    blimp_save_dir.mkdir(parents=True, exist_ok=True)
    
    # 下载每个现象
    success_count = 0
    for phenomenon in phenomena:
        try:
            print(f"Downloading phenomenon: {phenomenon}")
            # 加载数据集
            ds = load_dataset("nyu-mll/blimp", phenomenon, cache_dir='/scratch/yy5020/huggingface_cache/datasets')
            
            # 将数据集转换为列表并保存为JSONL
            items = []
            for item in ds['train']:
                items.append({
                    'sentence_good': item['sentence_good'],
                    'sentence_bad': item['sentence_bad'],
                    'field': item.get('field', ''),
                    'linguistic_feature': item.get('linguistic_feature', ''),
                    'simple_LM_method': item.get('simple_LM_method', True)
                })
            
            # 保存为JSONL
            output_path = blimp_save_dir / f"{phenomenon}.jsonl"
            with open(output_path, 'w') as f:
                for item in items:
                    f.write(json.dumps(item) + '\n')
            
            print(f"  Saved {len(items)} examples to {output_path}")
            success_count += 1
            
        except Exception as e:
            print(f"  Error downloading {phenomenon}: {e}")
            import traceback
            traceback.print_exc(file=sys.stdout)
        
        # 避免请求过于频繁
        time.sleep(0.2)
    
    # 创建元数据文件
    metadata = {
        "name": "BLiMP",
        "description": "Benchmark of Linguistic Minimal Pairs",
        "phenomena_downloaded": success_count,
        "total_phenomena": len(phenomena),
        "download_date": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    metadata_path = raw_dir / "blimp" / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"BLiMP dataset downloaded! Successfully downloaded {success_count}/{len(phenomena)} phenomena.")
    print(f"Metadata saved to {metadata_path}")

if __name__ == "__main__":
    print("Starting dataset downloads...")
    download_cola()
    download_blimp()
    print("All datasets downloaded successfully!")
