# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .collator import (
    KTODataCollatorWithPadding,
    MultiModalDataCollatorForSeq2Seq,
    PairwiseDataCollatorWithPadding,
    SFTDataCollatorWith4DAttentionMask,
)
from .data_utils import Role, split_dataset
from .loader import get_dataset
from .template import TEMPLATES, Template, get_template_and_fix_tokenizer
from typing import TYPE_CHECKING, Dict
from llamafactory.extras.logging import get_logger
from llamafactory.extras.template import Template
from .custom_dataset import get_custom_dataset

if TYPE_CHECKING:
    from torch.utils.data import Dataset

logger = get_logger(__name__)

def get_dataset(
    dataset_info: Dict,
    dataset_path: str,
    template: Template,
    **kwargs
) -> Dict[str, "Dataset"]:
    """
    根据数据集类型获取相应的数据集。

    Args:
        dataset_info (Dict): 数据集配置信息
        dataset_path (str): 数据集路径
        template (Template): 模板对象
        **kwargs: 其他参数

    Returns:
        Dict[str, Dataset]: 包含训练集的字典
    """
    dataset_type = dataset_info.get("type", "custom")
    
    if dataset_type == "custom":
        return get_custom_dataset(dataset_path, template, **kwargs)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

__all__ = [
    "TEMPLATES",
    "KTODataCollatorWithPadding",
    "MultiModalDataCollatorForSeq2Seq",
    "PairwiseDataCollatorWithPadding",
    "Role",
    "SFTDataCollatorWith4DAttentionMask",
    "Template",
    "get_dataset",
    "get_template_and_fix_tokenizer",
    "split_dataset",
]
