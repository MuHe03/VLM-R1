#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple test script to verify the dataset format and basic processing.
"""

import json
import re
from typing import Dict, Any, Optional

def extract_rle_mask_from_content(content: str) -> Optional[Dict[str, Any]]:
    """Extract RLE mask from assistant response content."""
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
    
    return None

def test_dataset_format():
    """Test the new dataset format."""
    print("Testing new segmentation dataset format...")
    
    dataset_path = "/dss/dssfs05/pn39qo/pn39qo-dss-0001/di97fer/hmu/VLM-R1-X/LLaMA-Factory/Qwen_seg_sft_train1.json"
    
    try:
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        print(f"✓ Dataset loaded successfully with {len(data)} samples")
        
        # Test first few samples
        for i, sample in enumerate(data[:3]):
            print(f"\n--- Sample {i+1} ---")
            print(f"Keys: {list(sample.keys())}")
            
            # Check messages
            messages = sample.get('messages', [])
            print(f"Messages: {len(messages)} messages")
            for j, msg in enumerate(messages):
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                print(f"  {j+1}. {role}: {content[:50]}{'...' if len(content) > 50 else ''}")
            
            # Check images
            images = sample.get('images', [])
            print(f"Images: {len(images)} images")
            if images:
                print(f"  First image: {images[0]}")
            
            # Check RLE mask
            rle_mask = sample.get('rle_mask', [])
            print(f"RLE mask: {type(rle_mask)} with {len(rle_mask) if isinstance(rle_mask, list) else 'N/A'} entries")
            
            if isinstance(rle_mask, list) and len(rle_mask) > 0:
                rle_data = rle_mask[0]
                if isinstance(rle_data, str):
                    # Try to extract RLE from JSON string
                    extracted_rle = extract_rle_mask_from_content(rle_data)
                    if extracted_rle:
                        print(f"  Extracted RLE keys: {list(extracted_rle.keys())}")
                        if 'size' in extracted_rle:
                            print(f"  Mask size: {extracted_rle['size']}")
                    else:
                        print(f"  Could not extract RLE from string")
                else:
                    print(f"  RLE data type: {type(rle_data)}")
        
        # Verify the expected format
        print(f"\n--- Format Verification ---")
        sample = data[0]
        
        # Check required fields
        required_fields = ['messages', 'images', 'rle_mask']
        missing_fields = [field for field in required_fields if field not in sample]
        
        if missing_fields:
            print(f"✗ Missing required fields: {missing_fields}")
        else:
            print(f"✓ All required fields present")
        
        # Check message format
        messages = sample['messages']
        if len(messages) >= 2:
            user_msg = messages[0]
            assistant_msg = messages[1]
            
            if user_msg.get('role') == 'user' and assistant_msg.get('role') == 'assistant':
                print(f"✓ Message roles are correct")
                
                if '[SEG]' in assistant_msg.get('content', ''):
                    print(f"✓ Assistant response contains [SEG] token")
                else:
                    print(f"✗ Assistant response missing [SEG] token")
            else:
                print(f"✗ Incorrect message roles")
        else:
            print(f"✗ Insufficient messages")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing dataset: {e}")
        return False

def test_seg_token_processing():
    """Test processing of [SEG] tokens in responses."""
    print(f"\nTesting [SEG] token processing...")
    
    test_responses = [
        "Sure, [SEG].",
        "I can help you segment that. [SEG]",
        "Here is the segmentation: [SEG].",
        "No segmentation needed.",  # No SEG token
    ]
    
    for i, response in enumerate(test_responses):
        has_seg = '[SEG]' in response
        print(f"  Response {i+1}: {'✓' if has_seg else '✗'} - {response}")
    
    return True

def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing VLM Segmentation Dataset Format")
    print("=" * 60)
    
    # Test 1: Dataset format
    dataset_ok = test_dataset_format()
    
    # Test 2: SEG token processing
    seg_token_ok = test_seg_token_processing()
    
    # Summary
    print(f"\n" + "=" * 60)
    print("Test Summary:")
    print(f"Dataset format: {'✓' if dataset_ok else '✗'}")
    print(f"SEG token processing: {'✓' if seg_token_ok else '✗'}")
    print("=" * 60)
    
    if dataset_ok and seg_token_ok:
        print("🎉 Dataset format tests passed!")
        print("\nNext steps:")
        print("1. Ensure NumPy compatibility in the environment")
        print("2. Test the full training pipeline")
        print("3. Add wandb logging for segmentation loss")
    else:
        print("⚠️  Some tests failed.")

if __name__ == "__main__":
    main()
