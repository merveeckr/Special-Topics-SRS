#!/usr/bin/env python3
"""
Performance test script for the optimized app.py
Run this to test the performance improvements before using the main app.
"""

import time
import sys
import os
import pandas as pd
import numpy as np

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_model_loading_performance():
    """Test model loading performance"""
    print("Testing model loading performance...")
    
    # Test local BERTürk model
    local_bert_path = "./BERTürk"
    if os.path.exists(local_bert_path):
        try:
            from transformers import AutoTokenizer
            import torch
            
            # Test tokenizer loading
            start_time = time.perf_counter()
            tokenizer = AutoTokenizer.from_pretrained(local_bert_path, local_files_only=True)
            tokenizer_time = time.perf_counter() - start_time
            
            print(f"✅ Local BERTürk tokenizer loaded in {tokenizer_time:.2f} seconds")
            
            # Test model loading (if available)
            if os.path.exists("best_model.pt"):
                from bil import BertBiLSTMCNN, LABEL_COLS
                start_time = time.perf_counter()
                model = BertBiLSTMCNN(bert_model_name=local_bert_path, num_labels=len(LABEL_COLS))
                model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                state = torch.load("best_model.pt", map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                model.load_state_dict(state, strict=False)
                model.eval()
                model_time = time.perf_counter() - start_time
                
                print(f"✅ Local BERTürk model loaded in {model_time:.2f} seconds")
            else:
                print("⚠️  No trained model found (best_model.pt)")
                
        except Exception as e:
            print(f"❌ Local BERTürk model loading failed: {e}")
    else:
        print(f"❌ Local BERTürk model not found: {local_bert_path}")
    
    # Test local Gemma model
    local_gemma_path = "./gemma"
    if os.path.exists(local_gemma_path):
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            start_time = time.perf_counter()
            tokenizer = AutoTokenizer.from_pretrained(local_gemma_path, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(local_gemma_path, local_files_only=True)
            gemma_time = time.perf_counter() - start_time
            
            print(f"✅ Local Gemma model loaded in {gemma_time:.2f} seconds")
            
        except Exception as e:
            print(f"❌ Local Gemma model loading failed: {e}")
    else:
        print(f"❌ Local Gemma model not found: {local_gemma_path}")

def test_batch_processing():
    """Test batch processing performance"""
    print("\nTesting batch processing performance...")
    
    try:
        # Create test data
        test_texts = [
            "Sistem kullanıcı girişi yapmalıdır",
            "Ürün stok durumu kontrol edilebilir olmalıdır",
            "Raporlar PDF formatında export edilebilmelidir"
        ] * 10  # 30 test texts
        
        from bil import MODEL_NAME, MAX_LEN, THRESH, LABEL_COLS
        from transformers import AutoTokenizer
        import torch
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
        
        # Test tokenization performance
        start_time = time.perf_counter()
        encodings = []
        for text in test_texts:
            enc = tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=MAX_LEN,
                return_tensors="pt",
            )
            encodings.append(enc)
        tokenization_time = time.perf_counter() - start_time
        
        print(f"✅ Tokenized {len(test_texts)} texts in {tokenization_time:.2f} seconds")
        print(f"   Average: {tokenization_time/len(test_texts)*1000:.1f} ms per text")
        
    except Exception as e:
        print(f"❌ Batch processing test failed: {e}")

def test_csv_loading():
    """Test CSV loading performance"""
    print("\nTesting CSV loading performance...")
    
    csv_files = ["short.csv", "yeni.csv"]
    
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            try:
                start_time = time.perf_counter()
                df = pd.read_csv(csv_file)
                load_time = time.perf_counter() - start_time
                
                print(f"✅ {csv_file}: {len(df)} rows loaded in {load_time:.2f} seconds")
                
                # Check required columns
                if "Requirement" in df.columns:
                    print(f"   ✅ 'Requirement' column found")
                else:
                    print(f"   ❌ 'Requirement' column missing")
                    
            except Exception as e:
                print(f"❌ Failed to load {csv_file}: {e}")
        else:
            print(f"⚠️  {csv_file} not found")

def main():
    """Run all performance tests"""
    print("🚀 Performance Test for Optimized App.py")
    print("=" * 50)
    
    test_csv_loading()
    test_model_loading_performance()
    test_batch_processing()
    
    print("\n" + "=" * 50)
    print("✅ Performance tests completed!")
    print("\nTips for better performance:")
    print("1. Use local model files when possible (set HF offline=True)")
    print("2. Keep batch sizes small (8-16) for memory efficiency")
    print("3. Use cached models in Streamlit session state")
    print("4. Monitor memory usage with large datasets")

if __name__ == "__main__":
    main()
