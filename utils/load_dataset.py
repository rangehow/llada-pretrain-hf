import datasets
import os
import json
from functools import wraps
from typing import Dict, Callable, Any, Optional
from pathlib import Path

# 全局数据集注册表
_DATASET_REGISTRY: Dict[str, Callable] = {}

# 数据集本地路径配置
_LOCAL_PATHS: Dict[str, str] = {}

def load_dataset_config(config_path: str = "diffusion/dataset_config.json"):
    """
    从配置文件加载数据集本地路径映射
    """
    global _LOCAL_PATHS
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                _LOCAL_PATHS = config.get('local_paths', {})
                print(f"已加载数据集配置: {len(_LOCAL_PATHS)} 个本地路径")
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            _LOCAL_PATHS = {}
    else:
        print(f"配置文件 {config_path} 不存在，将使用默认远程加载")
        _LOCAL_PATHS = {}

def register_dataset(remote_loader: Optional[Callable] = None):
    """
    数据集注册装饰器
    支持本地路径优先加载
    """
    def decorator(func: Callable) -> Callable:
        dataset_name = func.__name__
        
        if dataset_name in _DATASET_REGISTRY:
            raise ValueError(f"数据集 '{dataset_name}' 已经存在，请使用不同的函数名")
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            local_path = _LOCAL_PATHS.get(dataset_name)
            return func(local_path, *args, **kwargs)
        
        _DATASET_REGISTRY[dataset_name] = wrapper
        return wrapper
    
    return decorator

def get_dataset(name: str, *args, **kwargs) -> Any:
    """
    根据名称获取数据集
    """
    if name not in _DATASET_REGISTRY:
        available_datasets = list(_DATASET_REGISTRY.keys())
        raise ValueError(f"数据集 '{name}' 未找到。可用数据集: {available_datasets}")
    
    return _DATASET_REGISTRY[name](*args, **kwargs)

def list_datasets() -> list:
    """
    列出所有已注册的数据集名称
    """
    return list(_DATASET_REGISTRY.keys())

def get_local_path(dataset_name: str) -> Optional[str]:
    """
    获取数据集的本地路径
    """
    return _LOCAL_PATHS.get(dataset_name)

# 初始化时加载配置
load_dataset_config()

# 注册数据集
@register_dataset()
def ultra_fineweb(local_path):
    """加载 Ultra-FineWeb 数据集"""
    if local_path is not None and os.path.exists(local_path):
        print(f"从本地路径加载数据集 'ultra_fineweb': {local_path}")
        dataset = datasets.load_dataset(local_path,name='default',split='en[:10%]',num_proc=64)
    else:
        print("从远程加载数据集 'ultra_fineweb'")
        dataset = datasets.load_dataset("openbmb/Ultra-FineWeb",'en',num_proc=64)
    
    dataset = dataset.rename_column('content','text')
    return dataset


@register_dataset()
def finefineweb(local_path):
    """加载 FineFineWeb 训练数据集"""
    if local_path is not None and os.path.exists(local_path):
        print(f"从本地路径加载数据集 'finefineweb': {local_path}")
        dataset = datasets.load_dataset(local_path,num_proc=64)['train']
    else:
        print("从远程加载数据集 'finefineweb'")
        dataset = datasets.load_dataset("m-a-p/FineFineWeb-sample",num_proc=64)['train']
    
    return dataset

# <<< ADDED: 新增验证集加载函数 >>>
@register_dataset()
def finefineweb_validation(local_path):
    """加载 FineFineWeb 验证数据集"""
    if local_path is not None and os.path.exists(local_path):
        print(f"从本地路径加载数据集 'finefineweb_validation': {local_path}")
        # 假设本地路径也是一个标准的Hugging Face数据集目录
        dataset = datasets.load_dataset(local_path, num_proc=64,split='train[:50%]')
    else:
        print("从远程加载数据集 'finefineweb_validation'")
        # 加载指定的验证集
        dataset = datasets.load_dataset("m-a-p/FineFineWeb-validation", num_proc=64,split='train[:50%]')
    
    return dataset
# <<< END ADDED >>>


@register_dataset()
def filtered_finefineweb(local_path):
    """加载 filtered_finefineweb 数据集"""
    print(f"从本地路径加载数据集 'filtered_finefineweb': {local_path}")
    dataset = datasets.load_from_disk(local_path)
    return dataset


@register_dataset()
def debug(local_path):
    dataset = datasets.Dataset.from_dict({"text": ["Today is a really good day isn't it? Wanna go for a walk?","hello world"]})
    return dataset

@register_dataset()
def fineweb_10b(local_path):
    """加载 fineweb-edu-dedup-10b 数据集"""
    if local_path is not None and os.path.exists(local_path):
        print(f"从本地路径加载数据集 'fineweb_10b': {local_path}")
        try:
            return datasets.load_dataset(local_path,num_proc=64,verification_mode='no_checks')['train']
        except Exception as e:
            print(f"本地加载失败: {e}, 尝试远程加载")
    
    print("从远程加载数据集 'fineweb_10b'")
    return datasets.load_dataset("EleutherAI/fineweb-edu-dedup-10b")['train']


@register_dataset()
def common_crawl(local_path):
    """加载 Common Crawl 数据集"""
    if local_path is not None and os.path.exists(local_path):
        print(f"从本地路径加载数据集 'common_crawl': {local_path}")
        try:
            return datasets.load_from_disk(local_path)
        except Exception as e:
            print(f"本地加载失败: {e}, 尝试远程加载")
    
    print("从远程加载数据集 'common_crawl'")
    return datasets.load_dataset("common_crawl")

@register_dataset()
def wikipedia(local_path):
    """加载 Wikipedia 数据集"""
    if local_path is not None and os.path.exists(local_path):
        print(f"从本地路径加载数据集 'wikipedia': {local_path}")
        try:
            return datasets.load_from_disk(local_path)
        except Exception as e:
            print(f"本地加载失败: {e}, 尝试远程加载")
    
    print("从远程加载数据集 'wikipedia'")
    return datasets.load_dataset("wikipedia", "20220301.en")