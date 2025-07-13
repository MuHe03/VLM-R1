import json
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from PIL import Image
from datasets import Dataset, load_dataset
from torch.utils.data import Dataset as TorchDataset
import pycocotools.mask as mask_util

from llamafactory.extras.logging import get_logger
from llamafactory.data.template import Template

logger = get_logger(__name__)

def rle_to_mask(rle: Dict[str, Any]) -> np.ndarray:
    """
    将RLE格式转换为二值掩码。

    Args:
        rle (Dict[str, Any]): RLE编码的掩码，包含 'counts' 和 'size' 字段

    Returns:
        np.ndarray: 二值掩码
    """
    return mask_util.decode(rle).astype(np.int32)

def extract_rle_from_response(response: str) -> Optional[Dict[str, Any]]:
    """
    从助手响应中提取RLE信息。

    Args:
        response (str): 助手的响应文本

    Returns:
        Optional[Dict[str, Any]]: RLE信息，如果没有找到则返回None
    """
    try:
        # 提取JSON部分
        json_start = response.find('```json\n') + 7
        json_end = response.find('\n```', json_start)
        if json_start == -1 or json_end == -1:
            return None
        
        json_str = response[json_start:json_end]
        data = json.loads(json_str)
        
        # 提取RLE信息
        if isinstance(data, list) and len(data) > 0 and 'rle_mask' in data[0]:
            return data[0]['rle_mask']
    except:
        return None
    
    return None

def load_custom_dataset(
    dataset_path: str,
    template: Template,
    **kwargs
) -> Dataset:
    """
    加载自定义数据集。

    Args:
        dataset_path (str): 数据集文件路径
        template (Template): 模板对象
        **kwargs: 其他参数

    Returns:
        Dataset: 处理后的数据集
    """
    # 读取JSON文件
    with open(dataset_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # 处理数据
    processed_data = {
        "conversations": [],
        "images": [],
        "masks": []
    }

    for item in raw_data:
        # 处理对话
        conversation = []
        for msg in item["messages"]:
            conversation.append({
                "role": msg["role"],
                "content": msg["content"]
            })
            # 如果是assistant的回复，尝试提取RLE信息
            if msg["role"] == "assistant":
                rle = extract_rle_from_response(msg["content"])
                if rle is not None:
                    mask = rle_to_mask(rle)
                    processed_data["masks"].append(mask)
        
        processed_data["conversations"].append(conversation)
        processed_data["images"].append(item.get("images", []))

    # 创建数据集
    dataset = Dataset.from_dict(processed_data)
    
    return dataset

def get_custom_dataset(
    dataset_path: str,
    template: Template,
    **kwargs
) -> Dict[str, TorchDataset]:
    """
    获取处理后的自定义数据集。

    Args:
        dataset_path (str): 数据集文件路径
        template (Template): 模板对象
        **kwargs: 其他参数

    Returns:
        Dict[str, TorchDataset]: 包含训练集的字典
    """
    dataset = load_custom_dataset(dataset_path, template, **kwargs)
    return {"train": dataset} 