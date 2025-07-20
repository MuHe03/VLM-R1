#[DONE:dyzhou:14.07] 分割数据集加载器 - 处理Qwen_val.json格式数据
"""
Segmentation dataset loader for Qwen_val.json format
"""

import json
import os
from typing import Dict, List, Any, Optional
from datasets import Dataset
from PIL import Image
import numpy as np
from pycocotools import mask as maskUtils


class SegmentationDataset:
    """Dataset class for segmentation tasks"""
    
    def __init__(self, data_path: str, image_root: Optional[str] = None):
        """
        Initialize segmentation dataset
        
        Args:
            data_path: Path to Qwen_val.json file
            image_root: Root directory for images (optional)
        """
        self.data_path = data_path
        self.image_root = image_root
        self.data = self._load_data()
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load data from JSON file"""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        processed_data = []
        for item in data:
            processed_item = self._process_item(item)
            if processed_item is not None:
                processed_data.append(processed_item)
        
        return processed_data
    
    def _process_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single data item"""
        try:
            # Extract messages
            messages = item.get('messages', [])
            if len(messages) < 2:
                return None
            
            user_message = messages[0]
            assistant_message = messages[1]
            
            # Extract image path
            images = item.get('images', [])
            if not images:
                return None
            
            image_path = images[0]
            if self.image_root:
                image_path = os.path.join(self.image_root, os.path.basename(image_path))
            
            # Extract RLE mask from assistant message
            rle_mask = self._extract_rle_mask(assistant_message.get('content', ''))
            
            return {
                'image_path': image_path,
                'question': user_message.get('content', ''),
                'answer': assistant_message.get('content', ''),
                'rle_mask': rle_mask,
                'metadata': item.get('metadata', {})
            }
        
        except Exception as e:
            print(f"Error processing item: {e}")
            return None
    
    def _extract_rle_mask(self, content: str) -> Dict[str, Any]:
        """Extract RLE mask from content"""
        import re
        
        pattern = r'```json\s*(.*?)\s*```'
        json_match = re.search(pattern, content, re.DOTALL)
        
        if json_match:
            try:
                json_content = json_match.group(1).strip()
                data = json.loads(json_content)
                if isinstance(data, list) and len(data) > 0:
                    return data[0].get('rle_mask', {})
            except json.JSONDecodeError:
                pass
        
        return {}
    
    def rle_to_mask(self, rle: Dict[str, Any]) -> np.ndarray:
        """Convert RLE encoding to binary mask"""
        if 'counts' not in rle or 'size' not in rle:
            return np.array([])
        
        try:
            mask = maskUtils.decode(rle)
            return mask.astype(bool)
        except Exception as e:
            print(f"Error decoding RLE: {e}")
            return np.array([])
    
    def load_image(self, image_path: str) -> Optional[Image.Image]:
        """Load image from path"""
        try:
            if os.path.exists(image_path):
                return Image.open(image_path).convert('RGB')
            else:
                print(f"Image not found: {image_path}")
                return None
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]
    
    def to_huggingface_dataset(self) -> Dataset:
        """Convert to HuggingFace Dataset"""
        return Dataset.from_list(self.data)


def create_segmentation_dataset(data_path: str, image_root: Optional[str] = None) -> SegmentationDataset:
    """Factory function to create segmentation dataset"""
    return SegmentationDataset(data_path, image_root) 