"""
Model downloader utility for pre-downloading models at startup.
"""
import os
import sys
from pathlib import Path
from typing import Optional

def download_model_at_startup(model_name: str = "microsoft/DialoGPT-small") -> bool:
    """
    Download the model at startup to avoid delays during inference.
    
    Args:
        model_name: Hugging Face model name to download
        
    Returns:
        True if model is available, False otherwise
    """
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        print(f"🔄 Pre-downloading model: {model_name}")
        print("   This may take a few minutes on first run...")
        
        # Download tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"   ✅ Tokenizer downloaded")
        
        # Download model
        model = AutoModelForCausalLM.from_pretrained(model_name)
        print(f"   ✅ Model downloaded")
        
        # Test inference
        test_input = "Hello, how are you?"
        inputs = tokenizer(test_input, return_tensors="pt", truncation=True, max_length=50)
        
        with model.eval():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        print(f"   ✅ Test inference successful: '{response}'")
        
        print(f"🎉 Model {model_name} ready for use!")
        return True
        
    except ImportError:
        print("❌ Transformers library not available")
        return False
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        return False

def check_model_availability(model_name: str = "microsoft/DialoGPT-small") -> bool:
    """
    Check if model is already downloaded and available.
    
    Args:
        model_name: Hugging Face model name to check
        
    Returns:
        True if model is available, False otherwise
    """
    try:
        from transformers import AutoTokenizer
        
        # Try to load tokenizer (this will use cache if available)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return True
        
    except Exception:
        return False

def get_model_info(model_name: str = "microsoft/DialoGPT-small") -> dict:
    """
    Get information about the model.
    
    Args:
        model_name: Hugging Face model name
        
    Returns:
        Dictionary with model information
    """
    model_info = {
        "microsoft/DialoGPT-small": {
            "size": "~117MB",
            "description": "Tiny conversational model",
            "speed": "Very fast",
            "quality": "Good for basic tasks"
        },
        "distilgpt2": {
            "size": "~82MB", 
            "description": "Tiny GPT-style model",
            "speed": "Very fast",
            "quality": "Good for text generation"
        },
        "google/flan-t5-small": {
            "size": "~60MB",
            "description": "Tiny instruction-following model",
            "speed": "Very fast", 
            "quality": "Good for analysis tasks"
        }
    }
    
    return model_info.get(model_name, {
        "size": "Unknown",
        "description": "Custom model",
        "speed": "Unknown",
        "quality": "Unknown"
    })

if __name__ == "__main__":
    # Test the downloader
    model_name = "microsoft/DialoGPT-small"
    
    print("🔍 Checking model availability...")
    if check_model_availability(model_name):
        print(f"✅ Model {model_name} is already available")
    else:
        print(f"📥 Model {model_name} not found, downloading...")
        success = download_model_at_startup(model_name)
        if success:
            print("🎉 Download completed successfully!")
        else:
            print("❌ Download failed")
    
    # Show model info
    info = get_model_info(model_name)
    print(f"\n📊 Model Info:")
    print(f"   Size: {info['size']}")
    print(f"   Description: {info['description']}")
    print(f"   Speed: {info['speed']}")
    print(f"   Quality: {info['quality']}")
