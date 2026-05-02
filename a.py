import os
import sys
import time
import difflib
from typing import List, Dict

import numpy as np
import pandas as pd
import streamlit as st

# Python 3.13 guard (PyTorch henüz desteklemiyor olabilir)
if sys.version_info >= (3, 13):
    st.error("Python 3.13 ile PyTorch/Transformers henüz stabil değil. Lütfen Python 3.10–3.12 kullanın.")
    st.stop()

# Lazy import: torch/transformers/bil yalnızca guard sonrası
try:
    from transformers import AutoTokenizer
    from bil import (
        LABEL_COLS,
        TEXT_COL,
        MODEL_NAME,
        MAX_LEN,
        DEVICE,
        THRESH,
        load_gemma_model,
        generate_ai_suggestion,
    )
    from bench import BertCNNOnly
except Exception as import_err:
    st.error(f"Kütüphane import hatası: {import_err}. Lütfen Python sürümünüzü ve paket kurulumlarınızı kontrol edin.")
    st.stop()


@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer(model_name: str):
    """Optimized model loading with better caching and error handling"""
    # Validate model name
    if not model_name or not model_name.strip():
        raise ValueError("Model name cannot be empty")
    
    # Check if it's a local path
    is_local_path = os.path.exists(model_name) or model_name.startswith("./") or model_name.startswith("../")
    
    try:
        if is_local_path:
            # Try local files first
            st.info(f"Yerel model yükleniyor: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        else:
            # Try local files first, then download
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            except Exception:
                st.info(f"Yerel model bulunamadı, indiriliyor: {model_name}")
                tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        st.error(f"Tokenizör yüklenemedi: {e}")
        raise
    
    model = BertCNNOnly(bert_model_name=model_name, num_labels=len(LABEL_COLS))
    import torch  # local import to avoid top-level import on unsupported envs
    model.to(DEVICE)
    
    # Eğer eğitilmiş ağırlıklar varsa yükle
    if os.path.exists("best_model.pt"):
        try:
            state = torch.load("best_model.pt", map_location=DEVICE)
            model.load_state_dict(state, strict=False)
            st.info("Eğitilmiş model ağırlıkları yüklendi.")
        except Exception as e:
            st.warning(f"Model weights could not be loaded: {e}")
    
    model.eval()
    return model, tokenizer


def batch_predict(model: BertCNNOnly, tokenizer, texts: List[str], max_len: int, threshold: float) -> np.ndarray:
    """Optimized batch prediction with progress tracking"""
    predictions = []
    batch_size = 8  # Reduced batch size for better memory management
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i+batch_size]
        batch_num = i // batch_size + 1
        
        status_text.text(f"Processing batch {batch_num}/{total_batches}")
        progress_bar.progress(batch_num / total_batches)
        
        enc = tokenizer(
            chunk,
            truncation=True,
            padding=True,
            max_length=max_len,
            return_tensors="pt",
        )
        import torch  # local import
        with torch.no_grad():
            logits = model(enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE))
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= threshold).astype(int)
        predictions.append(preds)
    
    progress_bar.empty()
    status_text.empty()
    return np.vstack(predictions) if predictions else np.zeros((0, len(LABEL_COLS)), dtype=int)


def build_editable_frame(df: pd.DataFrame, preds: np.ndarray) -> pd.DataFrame:
    pred_df = pd.DataFrame(preds, columns=[f"AI_{c}" for c in LABEL_COLS])
    merged = pd.concat([df.reset_index(drop=True), pred_df], axis=1)
    # Kullanıcı işaretlemesi için sütunlar
    for c in LABEL_COLS:
        merged[f"User_{c}"] = merged.get(c, 0)
    return merged


st.set_page_config(
    page_title="Gereksinim Analizi", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Performance optimizations
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
if "model_cache" not in st.session_state:
    st.session_state.model_cache = {}
if "llm_cache" not in st.session_state:
    st.session_state.llm_cache = {}

def build_llm_prompt(requirement: str, missing: List[str]) -> str:
    missing_str = ", ".join(missing)
    return (
        "Aşağıdaki gereksinimi, eksik yönlerini gidererek TEK CÜMLE hâlinde yeniden yaz.\n"
        "- Girişi tekrar ETME, cümleyi YENİDEN kur.\n"
        "- Yalnızca şu formatta dön: İyileştirilmiş gereksinim: <cümle>\n"
        "- Türkçe yaz; belirsizlikten kaçın (net, ölçülebilir, doğrulanabilir).\n"
        "- Gerekirse kabul ölçütlerini cümle içine açık ve sayısal geçir.\n"
        "- Listeleme, açıklama, 'Giriş/Sonuç' gibi kalıplar kullanma.\n"
        "\nÖrnek:\n"
        "Eksikler: Unambiguous, Verifiable\n"
        "Girdi: Sistem hızlı olmalı.\n"
        "Çıktı: İyileştirilmiş gereksinim: Sistem, en az %99 isabetle 200 ms içinde cevap verecektir.\n"
        "\nGerçek görev:\n"
        f"Eksikler: {missing_str}\n"
        f"Girdi: {requirement}"
    )

def build_zeroshot_prompt(requirement: str, missing: List[str]) -> str:
    missing_str = ", ".join(missing)
    return (
        "Aşağıdaki gereksinimi, eksik yönlerini gidererek TEK CÜMLE olarak yeniden yaz.\n"
        "- Sadece şu formatta dön: İyileştirilmiş gereksinim: <cümle>\n"
        "- Türkçe yaz; belirsizlikten kaçın (net, ölçülebilir, doğrulanabilir).\n"
        "- Liste yapma, açıklama ekleme.\n"
        f"Eksikler: {missing_str}\n"
        f"Girdi: {requirement}"
    )

def build_fewshot_prompt(requirement: str, missing: List[str]) -> str:
    missing_str = ", ".join(missing)
    examples = (
        "Örnek 1\n"
        "Eksikler: Unambiguous, Verifiable, Complete\n"
        "Girdi: Bir ders, klinik olmayan veya klinik ders olabilir.\n"
        "Çıktı: İyileştirilmiş gereksinim: Kullanıcı, ders türünü 'klinik' veya 'klinik olmayan' olarak seçecektir; seçim form gönderiminden önce zorunludur ve klinik derslerde vaka oturumu en az 45 dakika sürecek, klinik olmayan derslerde ödev teslimi içerik yayımlandıktan sonra 24 saat içinde yapılacaktır.\n\n"
        "Örnek 2\n"
        "Eksikler: Unambiguous, Verifiable, Feasible\n"
        "Girdi: Ürün, tüm ortamlarda mevcut donanım üzerinde çalışmalıdır.\n"
        "Çıktı: İyileştirilmiş gereksinim: Ürün; Windows 10+, macOS 12+ ve Ubuntu 22.04 üzerinde Intel i5 sınıfı CPU ve 8 GB RAM bulunan donanımda kurulacak, GPU gerektirmeyecek ve temel ekranlar 800 ms içinde yanıt verecektir.\n\n"
    )
    return (
        "Aşağıdaki gereksinimi, eksik yönlerini gidererek TEK CÜMLE olarak yeniden yaz.\n"
        "- Yalnızca şu formatta dön: İyileştirilmiş gereksinim: <cümle>\n"
        "- Türkçe yaz; belirsizlikten kaçın (net, ölçülebilir, doğrulanabilir).\n"
        "- Listeleme veya açıklama ekleme.\n\n"
        + examples +
        "Gerçek görev\n"
        f"Eksikler: {missing_str}\n"
        f"Girdi: {requirement}"
    )

def build_strict_prompt(requirement: str, missing: List[str]) -> str:
    missing_str = ", ".join(missing)
    return (
        "Aşağıdaki gereksinimi, eksik yönlerini giderecek şekilde TEK CÜMLE olarak yeniden yaz.\n"
        "- Girişi tekrar etme; tamamen yeniden formüle et.\n"
        "- Yalnızca şu formatta dön: İyileştirilmiş gereksinim: <cümle>\n"
        "- Listeleme yapma, 'Giriş/Sonuç' kalıpları kullanma, emoji/nezaket ifadeleri ekleme.\n"
        "- Zorunlu unsurlar: (1) menü/gezinti/butonlardan en az ikisi, (2) erişilebilirlik ifadesi, (3) en az iki sayısal sınır (ms cinsinden tepkisellik ve yükleme).\n"
        "- Örnek: İyileştirilmiş gereksinim: Kullanıcı arayüzünde üst menü çubuğu, yan gezinme menüsü ve geri/ileri butonları yer alacak; tüm menü ve buton etiketleri erişilebilirlik standartlarına uygun olacaktır ve kullanıcı tıklamalarına verilen görsel geri bildirim 150 ms, ilgili sayfa yükleme süresi 800 ms’yi aşmayacaktır.\n"
        "\nGerçek görev:\n"
        f"Eksikler: {missing_str}\n"
        f"Girdi: {requirement}"
    )

def validate_requirement_output(text: str) -> bool:
    import re
    t = _normalize_text(text)
    if not t or "\n" in text:
        return False
    # required keywords
    need_ui = any(k in t for k in ["menü", "menuler", "menüler", "gezin", "navigasyon", "buton"])
    need_access = "erişilebilir" in t or "erişilebilirlik" in t
    # at least two numbers, and preferably ms
    nums = re.findall(r"\d+\s*(ms|milisaniye)?", text.lower())
    need_numbers = len(nums) >= 2
    return bool(need_ui and need_access and need_numbers)

def _normalize_text(s: str) -> str:
    return str(s).strip().lower().rstrip('.').replace('\n', ' ')

CLAUSE_BY_LABEL = {
    'Unambiguous': "belirsiz terimler kullanılmadan net olarak",
    'Verifiable': "ölçülebilir ve doğrulanabilir kabul kriterleri ile",
    'Complete': "girdi, işlem ve çıktıları tanımlanmış olarak",
    'Conforming': "kurumsal standart ve yönergelere uygun şekilde",
    'Correct': "alan tanımlarına ve iş kurallarına uygun olarak",
    'Feasible': "mevcut sistem kısıtları içinde uygulanabilir düzeyde",
    'Necessary': "iş hedefleri açısından gerekli olan kapsamda",
    'Singular': "tek bir amacı ifade edecek şekilde",
    'Appropriate': "hedef kullanıcı kitlesine uygun biçimde",
}

def rule_based_improvement(requirement: str, missing: List[str]) -> str:
    base = _normalize_text(requirement)
    clauses = [CLAUSE_BY_LABEL.get(m, "") for m in missing]
    clauses = [c for c in clauses if c]
    if clauses:
        clause_text = ", ".join(clauses)
        improved = f"İyileştirilmiş gereksinim: {base} ve {clause_text} olacaktır."
    else:
        improved = f"İyileştirilmiş gereksinim: {base} olacaktır."
    return improved

def load_llm_model(model_name: str, backend: str, offline: bool, use_cpu_only: bool = True, max_memory_gb: int = 4):
    """Optimized LLM model loading with session state caching and memory management"""
    cache_key = f"{model_name}_{backend}_{offline}_{use_cpu_only}_{max_memory_gb}"
    
    # Check if model is already cached in session state
    if cache_key in st.session_state.llm_cache:
        return st.session_state.llm_cache[cache_key]
    
    # Check model type and show appropriate messages
    if "gemma" in model_name.lower():
        #st.warning("⚠️ Gemma modeli çok büyük! Yükleme 10-15 dakika sürebilir...")
        #st.info("💡 Daha hızlı sonuç için Dialog modeli kullanın!")
        
        # Show progress indicator for Gemma
        progress_container = st.container()
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Simulate progress updates
            for i in range(100):
                progress_bar.progress(i + 1)
                if i < 20:
                    status_text.text("Model dosyaları okunuyor...")
                elif i < 60:
                    status_text.text("Model ağırlıkları yükleniyor...")
                elif i < 90:
                    status_text.text("Model optimize ediliyor...")
                else:
                    status_text.text("Model hazırlanıyor...")
                
                import time
                time.sleep(0.1)  # Small delay to show progress
            
            progress_container.empty()
    elif "gemma-small" in model_name.lower():
        st.success("⚡ Gemma Small modeli yükleniyor!")
        st.info("🚀 Bu model 2-3 dakikada hazır olacak.")
    elif "dialog" in model_name.lower():
        st.warning("⚠️ Dialog modeli sorunlu olabilir!")
        st.info("💡 Gemma Small veya online modelleri deneyin.")
    
    if backend == "HF":
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            import torch
            import gc
            
            # Import Gemma tokenizer if needed
            if "gemma" in model_name.lower():
                try:
                    from transformers import GemmaTokenizer
                except ImportError:
                    st.warning("GemmaTokenizer import edilemedi, AutoTokenizer kullanılacak.")
            
            # Clear GPU memory before loading
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            # Check if it's a local path
            is_local_path = os.path.exists(model_name) or model_name.startswith("./") or model_name.startswith("../")
            
            if is_local_path or offline:
                st.info(f"Yerel LLM modeli yükleniyor: {model_name}")
                
                # Aggressive memory optimization for large models like Gemma
                if use_cpu_only:
                    device_map = None  # Don't use device_map for CPU
                    torch_dtype = torch.float32
                    device = -1
                else:
                    device_map = None  # Simplified approach
                    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                    device = 0 if torch.cuda.is_available() else -1
                
                # Load tokenizer with error handling
                try:
                    if "gemma" in model_name.lower():
                        # Gemma models need special tokenizer handling
                        st.info("Gemma tokenizer yükleniyor...")
                        # Explicit check for SentencePiece to avoid confusing fallbacks
                        try:
                            import sentencepiece  # noqa: F401
                        except Exception:
                            st.error(
                                "GemmaTokenizer için 'sentencepiece' gereklidir. Lütfen 'pip install sentencepiece' komutunu çalıştırın ve uygulamayı yeniden başlatın."
                            )
                            raise RuntimeError("SentencePiece missing for Gemma tokenizer")
                        
                        # Try multiple approaches for Gemma tokenizer
                        tokenizer = None
                        
                        # Method 1: Try with use_fast=False (for Gemma models)
                        try:
                            if is_local_path:
                                tokenizer = AutoTokenizer.from_pretrained(
                                    model_name, 
                                    local_files_only=True,
                                    use_fast=False,
                                    trust_remote_code=True
                                )
                            else:
                                tokenizer = AutoTokenizer.from_pretrained(
                                    model_name, 
                                    local_files_only=False,
                                    use_fast=False,
                                    trust_remote_code=True
                                )
                            st.success("✅ Gemma tokenizer (orijinal) yüklendi!")
                        except Exception as e1:
                            st.warning(f"Orijinal tokenizer başarısız: {e1}")
                            
                            # Method 2: Try with use_fast=True
                            try:
                                if is_local_path:
                                    tokenizer = AutoTokenizer.from_pretrained(
                                        model_name, 
                                        local_files_only=True,
                                        use_fast=True,
                                        trust_remote_code=True
                                    )
                                else:
                                    tokenizer = AutoTokenizer.from_pretrained(
                                        model_name, 
                                        local_files_only=False,
                                        use_fast=True,
                                        trust_remote_code=True
                                    )
                                st.success("✅ Gemma tokenizer (fast) yüklendi!")
                            except Exception as e2:
                                st.warning(f"Fast tokenizer başarısız: {e2}")
                                
                                # Method 3: Use GPT2 tokenizer as fallback
                                st.warning("⚠️ Gemma tokenizer başarısız, GPT2 tokenizer kullanılıyor...")
                                st.info("💡 Bu durumda cevaplar farklı olabilir.")
                                tokenizer = AutoTokenizer.from_pretrained("gpt2")
                                tokenizer.pad_token = tokenizer.eos_token
                                st.success("GPT2 tokenizer (fallback) yüklendi.")
                        
                    else:
                        # Standard tokenizer loading for other models (LLaMA, Phi, GPT2, etc.)
                        if is_local_path:
                            tokenizer = AutoTokenizer.from_pretrained(
                                model_name, 
                                local_files_only=True,
                                use_fast=True,
                                trust_remote_code=True
                            )
                        else:
                            tokenizer = AutoTokenizer.from_pretrained(
                                model_name, 
                                local_files_only=False,
                                use_fast=True,
                                trust_remote_code=True
                            )
                    
                    # Add padding token if not exists
                    if tokenizer.pad_token is None:
                        tokenizer.pad_token = tokenizer.eos_token
                        
                except Exception as tokenizer_error:
                    st.error(f"Tokenizer yüklenemedi: {tokenizer_error}")
                    raise
                
                # Optimize loading based on model type
                try:
                    if "roberta" in model_name.lower():
                        st.error("RoBERTa üretim için uygun değil (CausalLM değil). Lütfen bir CausalLM seçin (Gemma, LLaMA, Phi, GPT-2 ailesi).")
                        raise RuntimeError("RoBERTa is not a generative CausalLM")
                    elif "llama" in model_name.lower() or "llama" in os.path.basename(model_name).lower():
                        st.info("LLaMA ailesi modeli yükleniyor...")
                        model = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            local_files_only=is_local_path,
                            torch_dtype=torch_dtype,
                            low_cpu_mem_usage=True,
                            trust_remote_code=True,
                            device_map="auto" if not use_cpu_only else None,
                            max_memory={0: f"{max_memory_gb}GB"} if torch.cuda.is_available() and not use_cpu_only else None,
                            offload_folder="./temp_offload" if use_cpu_only else None
                        )
                    elif "phi" in model_name.lower() or "phi" in os.path.basename(model_name).lower():
                        st.info("Phi ailesi modeli yükleniyor...")
                        model = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            local_files_only=is_local_path,
                            torch_dtype=torch_dtype,
                            low_cpu_mem_usage=True,
                            trust_remote_code=True,
                            device_map="auto" if not use_cpu_only else None,
                            max_memory={0: f"{max_memory_gb}GB"} if torch.cuda.is_available() and not use_cpu_only else None,
                            offload_folder="./temp_offload" if use_cpu_only else None
                        )
                    elif "gemma-small" in model_name.lower():
                        # Gemma Small is optimized for better performance
                        st.info("Gemma Small modeli yükleniyor...")
                        try:
                            # Ensure CPU offload folder exists (Windows pagefile issues workaround)
                            if use_cpu_only:
                                try:
                                    os.makedirs("./temp_offload", exist_ok=True)
                                except Exception:
                                    pass
                            # Try with trust_remote_code first
                            model = AutoModelForCausalLM.from_pretrained(
                                model_name,
                                local_files_only=is_local_path,
                                torch_dtype=torch_dtype,
                                low_cpu_mem_usage=True,
                                trust_remote_code=True,
                                device_map="auto" if not use_cpu_only else None,
                                max_memory={0: f"{max_memory_gb}GB"} if torch.cuda.is_available() and not use_cpu_only else None,
                                offload_folder="./temp_offload" if use_cpu_only else None
                            )
                            st.success("Gemma Small modeli başarıyla yüklendi!")
                        except Exception as gemma_error:
                            st.warning(f"Gemma Small yüklenemedi: {gemma_error}")
                            st.info("Fallback model deneniyor...")
                            raise
                    elif "dialog" in model_name.lower():
                        # Dialog model is actually a GPT-2 model, load it directly
                        st.info("Dialog modeli GPT-2 tabanlı, yükleniyor...")
                        model = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            local_files_only=is_local_path,
                            torch_dtype=torch_dtype,
                            trust_remote_code=True
                        )
                    else:
                        # Ultra-aggressive memory optimization for large models like Gemma
                        model = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            local_files_only=is_local_path,
                            torch_dtype=torch_dtype,
                            low_cpu_mem_usage=True,
                            trust_remote_code=True,
                            device_map="auto" if not use_cpu_only else None,
                            max_memory={0: f"{max_memory_gb}GB"} if torch.cuda.is_available() and not use_cpu_only else None,
                            load_in_8bit=False,  # Disable 8-bit loading for compatibility
                            load_in_4bit=False   # Disable 4-bit loading for compatibility
                        )
                except Exception as model_error:
                    st.error(f"Model yüklenemedi: {model_error}")
                    raise
                
                # Move to device manually
                if device >= 0:
                    model = model.to(device)
            else:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
                    model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True)
                except Exception:
                    st.info(f"Yerel model bulunamadı, indiriliyor: {model_name}")
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = AutoModelForCausalLM.from_pretrained(model_name)
            
            generator = pipeline(
                "text-generation", 
                model=model, 
                tokenizer=tokenizer, 
                device=device, 
                return_full_text=False,
                torch_dtype=torch_dtype
            )
            st.session_state.llm_cache[cache_key] = generator
            return generator
        except Exception as e:
            st.error(f"HuggingFace model loading failed: {e}")
            
            # Try fallback to smaller model if any model fails
            st.warning(f"{model_name} modeli yüklenemedi! Daha küçük bir model deneniyor...")
            try:
                fallback_model = "microsoft/DialoGPT-small"
                st.info(f"Fallback model yükleniyor: {fallback_model}")
                
                tokenizer = AutoTokenizer.from_pretrained(fallback_model)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                model = AutoModelForCausalLM.from_pretrained(fallback_model)
                
                generator = pipeline(
                    "text-generation", 
                    model=model, 
                    tokenizer=tokenizer, 
                    device=-1,  # CPU only for fallback
                    return_full_text=False,
                    torch_dtype=torch.float32
                )
                
                # Cache the fallback model
                fallback_key = f"fallback_{fallback_model}"
                st.session_state.llm_cache[fallback_key] = generator
                st.success(f"✅ Fallback model yüklendi: {fallback_model}")
                return generator
                
            except Exception as fallback_error:
                st.error(f"Fallback model de yüklenemedi: {fallback_error}")
            
            st.warning("Tüm modeller yüklenemedi! Kural tabanlı öneri kullanılacak.")
            return None
    else:  # GGUF
        try:
            from llama_cpp import Llama
            model = Llama(
                model_path=model_name, 
                n_ctx=256,  # Very small context for memory
                n_threads=1,  # Single thread for memory
                n_gpu_layers=0,  # Force CPU usage
                verbose=False
            )
            st.session_state.llm_cache[cache_key] = model
            return model
        except Exception as e:
            st.error(f"GGUF model loading failed: {e}")
            return None

def enforce_improvement(requirement: str, generated: str, missing: List[str], similarity_threshold: float = 0.9) -> str:
    text = generated.strip()
    low = text.lower()
    if "iyileştirilmiş gereksinim" in low:
        try:
            start = low.index("iyileştirilmiş gereksinim")
            text = text[start:].split("\n",1)[0]
            if ':' in text:
                text = text.split(':',1)[1].strip()
        except Exception:
            pass
    text = text.split("\n",1)[0].strip().strip('"').strip("'")
    r_base = _normalize_text(requirement)
    r_out = _normalize_text(text)
    sim = difflib.SequenceMatcher(None, r_base, r_out).ratio()
    if sim >= similarity_threshold or len(r_out) <= len(r_base):
        return rule_based_improvement(requirement, missing)
    return f"İyileştirilmiş gereksinim: {text.rstrip('.')}." if not text.lower().startswith("iyileştirilmiş gereksinim") else text

def select_best_candidate(candidates: List[str], requirement: str, missing: List[str]) -> str:
    """Choose the candidate that is least similar to input but still coherent and includes cues for missing aspects."""
    requirement_norm = _normalize_text(requirement)
    # Build simple keyword cues from missing labels
    cues = [CLAUSE_BY_LABEL[m] for m in missing if m in CLAUSE_BY_LABEL]
    best = None
    best_score = -1e9
    for cand in candidates:
        text = str(cand or "").strip().split("\n",1)[0]
        out_norm = _normalize_text(text)
        sim = difflib.SequenceMatcher(None, requirement_norm, out_norm).ratio()
        # Features
        length_bonus = min(len(out_norm) - len(requirement_norm), 80) / 80.0  # encourage adding info
        cue_bonus = 0.0
        low = out_norm
        for cue in cues:
            for token in str(cue).split():
                if token and token.lower() in low:
                    cue_bonus += 0.02
        # Penalize starting with "giriş:"/"sonuç:" style
        penalty = -0.2 if (low.startswith("giriş:") or low.startswith("sonuç:")) else 0.0
        score = (-sim) + 0.3*length_bonus + cue_bonus + penalty
        if score > best_score:
            best_score = score
            best = text
    return best or (candidates[0] if candidates else "")

def _sanitize_llm_text(raw: str) -> str:
    import re
    s = str(raw or "")
    # Remove common conversational prefixes and tags
    s = s.replace("<end_of_turn>", " ")
    for prefix in ("giriş:", "girış:", "sonuç:", "cevap:"):
        if s.strip().lower().startswith(prefix):
            s = s.split(":", 1)[-1].strip()
    # Remove emojis and non-text symbols
    emoji_pattern = re.compile(r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\u2600-\u26FF\u2700-\u27BF]", flags=re.UNICODE)
    s = emoji_pattern.sub("", s)
    # Remove filler phrases
    fillers = ["umarız", "lütfen", "yardım ederiz", "teşekkür", "rica", "memnuniyetle"]
    low = s.lower()
    for f in fillers:
        if f in low:
            s = re.sub(f, "", s, flags=re.IGNORECASE)
    s = s.strip().strip('"').strip("'").strip()
    # One sentence
    s = s.split("\n", 1)[0].strip()
    return s.rstrip('.')

def _is_output_compliant(text: str, missing: List[str]) -> bool:
    import re
    t = _normalize_text(text)
    if len(t) < 30:
        return False
    # No emojis or friendly tones
    if any(token in t for token in ("umarız", "lütfen", "yardım ederiz", "😊", "🙂", "teşekkür")):
        return False
    # If verifiable is missing, require a number
    if any(m.lower() == "verifiable" for m in missing):
        if not re.search(r"\d", t):
            return False
    # Avoid vague words if Unambiguous missing
    vague = ["hızlı", "kolay", "mümkün", "gerekli düzeyde", "uygun"]
    if any(m.lower() == "unambiguous" for m in missing):
        if any(v in t for v in vague):
            return False
    return True
st.title("Gereksinim Analizi: BERTürk + Gemma Öneri")

with st.sidebar:
    st.markdown("**Model Ayarları**")
    # Use local BERTürk model
    local_bert_path = "./BERTürk"
    model_name = st.text_input("BERT Model", value=local_bert_path)
    
    # Check if local model exists
    if os.path.exists(local_bert_path):
        st.success(f"✅ Yerel BERT modeli bulundu: {local_bert_path}")
    else:
        st.warning(f"⚠️ Yerel BERT modeli bulunamadı: {local_bert_path}")
        model_name = MODEL_NAME
        st.info("HuggingFace modeli kullanılacak.")
    
    st.caption("Sınıflandırıcı: BERTürk + CNN  |  F1=0.9907  Acc=0.8950")
    threshold = st.slider("Eşik (sigmoid)", min_value=0.05, max_value=0.95, value=float(THRESH), step=0.05)
    st.divider()
    st.markdown("**LLM Ayarları**")
    llm_backend = st.selectbox("LLM Backend", options=["HF", "GGUF"], index=0)
    llm_offline = st.checkbox("HF offline", value=True)
    
    # Check for local models
    local_gemma_path = "./gemma"
    local_gemma_small_path = "./gemma-small"
    local_dialog_path = "./dialog"
    local_llama_path = "./llama"
    local_phi_path = "./phi"
    local_roberta_path = "./roberta"
    
    model_options = []
    
    # Add online small models first (most reliable)
    model_options.extend(["microsoft/DialoGPT-small", "distilgpt2"])
    
    # Add Gemma Small (with special handling)
    if os.path.exists(local_gemma_small_path):
        st.success(f"✅ Yerel Gemma Small modeli bulundu: {local_gemma_small_path}")
        st.info("⚡ Bu model orta boyutta ve hızlı!")
        st.warning("⚠️ Gemma tokenizer sorunu için özel çözüm uygulanacak.")
        model_options.append(local_gemma_small_path)
    
    # Skip dialog model for now (it has issues)
    # if os.path.exists(local_dialog_path):
    #     st.warning(f"⚠️ Yerel Dialog modeli bulundu: {local_dialog_path}")
    #     st.warning("⚠️ Bu model sorunlu olabilir, test edin.")
    #     model_options.append(local_dialog_path)
    
    # Add Gemma last (slowest)
    if os.path.exists(local_gemma_path):
        st.warning(f"⚠️ Yerel Gemma modeli bulundu: {local_gemma_path}")
        st.warning("⚠️ Gemma çok büyük! 10-15 dakika sürebilir.")
        model_options.append(local_gemma_path)

    # Add LLaMA family (HF format)
    if os.path.exists(local_llama_path):
        st.info(f"🔹 Yerel LLaMA modeli bulundu: {local_llama_path}")
        model_options.append(local_llama_path)

    # Add Phi family (HF format)
    if os.path.exists(local_phi_path):
        st.info(f"🔹 Yerel Phi modeli bulundu: {local_phi_path}")
        model_options.append(local_phi_path)

    # Add RoBERTa (not supported for generation)
    if os.path.exists(local_roberta_path):
        st.warning(f"⚠️ Yerel RoBERTa modeli bulundu: {local_roberta_path}")
        st.caption("RoBERTa üretim (text-generation) için uygun değildir; seçimde uyarı verilecektir.")
        model_options.append(local_roberta_path)
    
    default_llm_models = ";".join(model_options)
    
    llm_models_input = st.text_input("LLM modelleri (; ile)", value=default_llm_models)
    llm_models = [m.strip() for m in llm_models_input.split(";") if m.strip()]
    active_llm = st.selectbox("Kullanılacak LLM", llm_models) if llm_models else None
    
    # Bellek optimizasyonu seçenekleri
    st.markdown("**Bellek Optimizasyonu**")
    use_cpu_only = st.checkbox("Sadece CPU kullan (GPU bellek tasarrufu)", value=True)
    max_memory_gb = st.slider("Maksimum GPU belleği (GB)", min_value=1, max_value=16, value=4)
    st.markdown("**LLM Üretim Ayarları**")
    gen_temperature = st.slider("Sıcaklık (temperature)", min_value=0.1, max_value=1.0, value=0.3 if (active_llm and "gemma" in active_llm.lower()) else 0.1, step=0.05)
    gen_top_p = st.slider("Top-p", min_value=0.5, max_value=1.0, value=0.8 if (active_llm and "gemma" in active_llm.lower()) else 0.9, step=0.05)
    sim_threshold = st.slider("Benzerlik eşiği (LLM→kural geçişi)", min_value=0.7, max_value=0.95, value=0.9, step=0.01)
    force_llm_output = st.checkbox("LLM çıktısını her zaman kullan (normalize etme)", value=True)
    strict_retry = st.checkbox("Benzer çıkarsa bir kez daha sıkı uyarıyla dene", value=True)
    k_candidates = st.slider("Aday sayısı (en iyisini seç)", min_value=1, max_value=5, value=3)
    show_prompt = st.checkbox("Gönderilen promptu göster", value=False)
    hard_constraints = st.checkbox("Sıkı doğrulama (3 denemeye kadar)", value=True)
    prompt_mode = st.selectbox("Prompt modu", options=["Strict", "Default", "Zero-shot", "Few-shot"], index=0)
    
    st.divider()
    st.markdown("**Performans Bilgileri**")
    if st.session_state.model_loaded:
        st.success("✅ BERT modeli yüklendi")
    else:
        st.info("⏳ BERT modeli yüklenmedi")
    
    if st.session_state.llm_cache:
        st.success(f"✅ {len(st.session_state.llm_cache)} LLM modeli önbellekte")
    else:
        st.info("⏳ LLM modeli önbellekte değil")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Önbelleği Temizle"):
            st.session_state.model_cache.clear()
            st.session_state.llm_cache.clear()
            st.session_state.model_loaded = False
            st.rerun()
    
    with col2:
        if st.button("Belleği Temizle"):
            import torch
            import gc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            st.success("GPU/CPU belleği temizlendi!")

uploaded = st.file_uploader("CSV yükleyin", type=["csv"])

if uploaded is None:
    st.write("CSV yükleyin ve analiz edin.")
    st.stop()

if uploaded is not None:
    df = pd.read_csv(uploaded)
    if TEXT_COL not in df.columns:
        st.error(f"CSV içinde '{TEXT_COL}' kolonu bulunamadı.")
        st.stop()
    # Eksikse label kolonları ekle
    for c in LABEL_COLS:
        if c not in df.columns:
            df[c] = 0

    st.success(f"Yüklendi: {uploaded.name}, satır: {len(df)}")

    # Check if model is already loaded in session state
    if not st.session_state.model_loaded or model_name not in st.session_state.model_cache:
        try:
            with st.spinner("Model yükleniyor..."):
                model, tokenizer = load_model_and_tokenizer(model_name)
                st.session_state.model_cache[model_name] = (model, tokenizer)
                st.session_state.model_loaded = True
                st.success(f"Model başarıyla yüklendi: {model_name}")
        except Exception as e:
            st.error(f"Model yüklenemedi: {e}")
            st.info("Varsayılan model kullanılıyor...")
            model_name = MODEL_NAME
            model, tokenizer = load_model_and_tokenizer(model_name)
            st.session_state.model_cache[model_name] = (model, tokenizer)
            st.session_state.model_loaded = True
    else:
        model, tokenizer = st.session_state.model_cache[model_name]

    texts = df[TEXT_COL].fillna("").astype(str).tolist()
    with st.spinner("Tahmin yapılıyor..."):
        preds = batch_predict(model, tokenizer, texts, MAX_LEN, threshold)

    work = build_editable_frame(df[[TEXT_COL] + LABEL_COLS], preds)
    # Ensure checkbox-compatible dtypes (bool) for editor
    try:
        work[TEXT_COL] = work[TEXT_COL].astype(str)
        for c in LABEL_COLS:
            if c in work.columns:
                work[c] = work[c].fillna(0).astype(int).astype(bool)
            ai_c = f"AI_{c}"
            if ai_c in work.columns:
                work[ai_c] = work[ai_c].fillna(0).astype(int).astype(bool)
            user_c = f"User_{c}"
            if user_c in work.columns:
                work[user_c] = work[user_c].fillna(0).astype(int).astype(bool)
    except Exception:
        pass

    st.subheader("Sonuçlar")
    ai_cols = [f"AI_{c}" for c in LABEL_COLS]
    user_cols = [f"User_{c}" for c in LABEL_COLS]

    left, right = st.columns(2)
    with left:
        st.markdown("**AI tahminleri**")
        st.dataframe(
            work[[TEXT_COL] + ai_cols],
            use_container_width=True,
            height=500,
        )
    with right:
        st.markdown("**Kullanıcı işaretlemeleri**")
        user_view = work[[TEXT_COL] + user_cols].copy()
        edited_user = st.data_editor(
            user_view,
            num_rows="dynamic",
            use_container_width=True,
            height=500,
            column_config={
                TEXT_COL: st.column_config.TextColumn("Gereksinim", width=400),
                **{uc: st.column_config.CheckboxColumn(uc.replace("User_", "Kullanıcı ")) for uc in user_cols},
            },
        )

    # Per-row agreement
    try:
        # Coerce user-edited values to booleans robustly
        for uc in user_cols:
            if uc in edited_user.columns:
                edited_user[uc] = edited_user[uc].apply(
                    lambda v: bool(v) if isinstance(v, (bool, np.bool_, int, np.integer)) else str(v).strip().lower() in ("1","true","yes","on","x")
                )
        ai_mat = work[ai_cols].astype(bool).to_numpy()
        user_mat = edited_user[user_cols].astype(bool).to_numpy()
        agree_ratio = (ai_mat == user_mat).mean(axis=1)
        summary = pd.DataFrame({
            TEXT_COL: work[TEXT_COL].astype(str).str.slice(0, 80) + "...",
            "Uyum_orani": np.round(agree_ratio, 3),
            "Uyum_sayisi": (ai_mat == user_mat).sum(axis=1),
            "Toplam_label": ai_mat.shape[1]
        })
        st.markdown("**AI vs Kullanıcı Uyum Özeti**")
        st.dataframe(summary, use_container_width=True, height=240)
    except Exception as e:
        st.warning(f"Uyum hesabı yapılamadı: {e}")

    # Birleştirilmiş çerçeve (AI + User) diğer adımlar için
    merged = work[[TEXT_COL] + ai_cols].copy()
    for uc in user_cols:
        merged[uc] = edited_user[uc].astype(bool)

    # Eksik etiketlerin çıkarımı
    def row_missing(row) -> List[str]:
        return [c for c in LABEL_COLS if int(row.get(f"AI_{c}", 0)) == 0]

    merged["Eksikler_AI"] = merged.apply(row_missing, axis=1)

    st.download_button(
        label="CSV indir",
        data=merged.to_csv(index=False).encode("utf-8"),
        file_name="gereksinim_sonuclari.csv",
        mime="text/csv",
    )

    st.subheader("LLM ile Öneri Üretimi")
    with st.expander("Satır seçerek öneri üret"):
        sel_idx = st.number_input("Satır index", min_value=0, max_value=len(merged)-1, value=0, step=1)
        if st.button("Seçili satıra öneri üret"):
            req_text = merged.iloc[sel_idx][TEXT_COL]
            miss = [c for c in LABEL_COLS if int(merged.iloc[sel_idx].get(f"AI_{c}", 0)) == 0]
            if not miss:
                st.warning("AI'ya göre eksik yok.")
            else:
                try:
                    if prompt_mode == "Strict":
                        prompt = build_strict_prompt(req_text, miss)
                    elif prompt_mode == "Zero-shot":
                        prompt = build_zeroshot_prompt(req_text, miss)
                    elif prompt_mode == "Few-shot":
                        prompt = build_fewshot_prompt(req_text, miss)
                    else:
                        prompt = build_llm_prompt(req_text, miss)
                    if active_llm is None:
                        st.error("LLM modeli seçin.")
                    else:
                        # Use optimized cached model loading
                        llm_model = load_llm_model(active_llm, llm_backend, llm_offline, use_cpu_only, max_memory_gb)
                        if llm_model is None:
                            st.error("LLM modeli yüklenemedi.")
                        else:
                            with st.spinner("LLM önerisi oluşturuluyor..."):
                                t0 = time.perf_counter()
                                
                                if llm_backend == "HF":
                                    # Optimize parameters based on model type
                                    if "gemma" in active_llm.lower():
                                        # Gemma models work better with different parameters
                                        out = llm_model(
                                            prompt, 
                                            max_new_tokens=48,  # Slightly more for Gemma
                                            do_sample=True, 
                                            num_beams=1,
                                            temperature=float(gen_temperature),
                                            top_p=float(gen_top_p),
                                            pad_token_id=llm_model.tokenizer.eos_token_id
                                        )
                                    else:
                                        # Standard parameters for other models
                                        out = llm_model(
                                            prompt, 
                                            max_new_tokens=32,  # Very small for memory efficiency
                                            do_sample=gen_temperature > 0.15,
                                            num_beams=1,  # Single beam for memory
                                            temperature=float(gen_temperature),
                                            pad_token_id=llm_model.tokenizer.eos_token_id
                                        )
                                    # Multi-candidate sampling for diversity
                                    gens = []
                                    for _ in range(int(max(1, k_candidates))):
                                        o = llm_model(
                                            prompt,
                                            max_new_tokens=64,
                                            min_new_tokens=24,
                                            do_sample=True,
                                            num_beams=1,
                                            temperature=float(gen_temperature),
                                            top_p=float(gen_top_p),
                                            no_repeat_ngram_size=2,
                                            repetition_penalty=1.15,
                                            pad_token_id=llm_model.tokenizer.eos_token_id
                                        )
                                        gens.append(o[0].get('generated_text','').strip())
                                    text = select_best_candidate(gens, req_text, miss)
                                    # Strict retry if too similar to input
                                    if strict_retry:
                                        r_base = _normalize_text(req_text)
                                        r_out = _normalize_text(text)
                                        if difflib.SequenceMatcher(None, r_base, r_out).ratio() >= float(sim_threshold):
                                            retry_prompt = (
                                                build_llm_prompt(req_text, miss)
                                                + "\n\nNot: Yukarıdaki cümleyi tekrar etme; yeni, daha NET, ÖLÇÜLEBİLİR ve DOĞRULANABİLİR tek cümle üret."
                                            )
                                            out2 = llm_model(
                                                retry_prompt, 
                                                max_new_tokens=48,
                                                do_sample=True,
                                                num_beams=1,
                                                temperature=float(gen_temperature),
                                                top_p=float(gen_top_p),
                                                pad_token_id=llm_model.tokenizer.eos_token_id
                                            )
                                            text = out2[0].get('generated_text','').strip()
                                else:  # GGUF
                                    gens = []
                                    for _ in range(int(max(1, k_candidates))):
                                        o = llm_model.create_completion(
                                            prompt=prompt, 
                                            max_tokens=64,
                                            temperature=float(gen_temperature), 
                                            top_p=float(gen_top_p),
                                            repeat_penalty=1.1
                                        )
                                        gens.append(o["choices"][0]["text"].strip())
                                    text = select_best_candidate(gens, req_text, miss)
                                    if strict_retry:
                                        r_base = _normalize_text(req_text)
                                        r_out = _normalize_text(text)
                                        if difflib.SequenceMatcher(None, r_base, r_out).ratio() >= float(sim_threshold):
                                            retry_prompt = (
                                                build_llm_prompt(req_text, miss)
                                                + "\n\nNot: Yukarıdaki cümleyi tekrar etme; yeni, daha NET, ÖLÇÜLEBİLİR ve DOĞRULANABİLİR tek cümle üret."
                                            )
                                            out2 = llm_model.create_completion(
                                                prompt=retry_prompt,
                                                max_tokens=48,
                                                temperature=float(gen_temperature),
                                                top_p=float(gen_top_p),
                                                repeat_penalty=1.1
                                            )
                                            text = out2["choices"][0]["text"].strip()
                                
                                dt = time.perf_counter() - t0
                                # Cleaning helper for forced LLM output
                                def _clean_llm_text(raw: str) -> str:
                                    return _sanitize_llm_text(raw)

                                # Compare with rule-based to detect normalization/fallback
                                rb_text = rule_based_improvement(req_text, miss)
                                if force_llm_output:
                                    # Hard validation loop (up to 3 tries)
                                    attempt = 0
                                    best_text = text
                                    while hard_constraints and attempt < 3:
                                        cleaned = _sanitize_llm_text(best_text)
                                        if validate_requirement_output(cleaned):
                                            break
                                        attempt += 1
                                        strict = build_strict_prompt(req_text, miss)
                                        o = llm_model(
                                            strict,
                                            max_new_tokens=64,
                                            min_new_tokens=24,
                                            do_sample=True,
                                            num_beams=1,
                                            temperature=float(gen_temperature),
                                            top_p=float(gen_top_p),
                                            no_repeat_ngram_size=2,
                                            repetition_penalty=1.15,
                                            pad_token_id=llm_model.tokenizer.eos_token_id
                                        )
                                        best_text = o[0].get('generated_text','').strip()
                                    cleaned = _sanitize_llm_text(best_text)
                                    if not validate_requirement_output(cleaned):
                                        # fallback to best candidate selection
                                        cleaned = _sanitize_llm_text(select_best_candidate(gens, req_text, miss))
                                    final_text = f"İyileştirilmiş gereksinim: {cleaned}."
                                else:
                                    final_text = enforce_improvement(req_text, text, miss, similarity_threshold=float(sim_threshold))
                                if show_prompt:
                                    with st.expander("Gönderilen prompt"):
                                        st.code(prompt)
                                st.text_area("AI Önerisi", value=final_text, height=200)
                                if not force_llm_output and final_text.strip() == rb_text.strip():
                                    st.caption("Kaynak: LLM çıktısi normalleştirildi veya kural tabanlı metne yakın olduğu için kural tabanlı metin kullanıldı.")
                                else:
                                    st.caption("Kaynak: LLM (üretilen metin korundu)")
                                with st.expander("Ham LLM çıktısı"):
                                    st.write(text)
                                st.info(f"Süre: {dt:.2f} sn")
                except Exception as e:
                    st.error(f"LLM çalıştırılamadı: {e}")
                    # Fallback to rule-based improvement
                    final_text = rule_based_improvement(req_text, miss)
                    st.text_area("AI Önerisi (Kural tabanlı)", value=final_text, height=200)

    st.subheader("Tekil Gereksinim Analizi ve Öneri")
    single_req = st.text_area("Gereksinimi girin", placeholder="Gereksinimi buraya yapıştırın", height=120)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Eksikleri Analiz Et") and single_req.strip():
            enc = tokenizer(
                [single_req],
                truncation=True,
                padding=True,
                max_length=MAX_LEN,
                return_tensors="pt",
            )
            import torch  # local import
            with torch.no_grad():
                logits = model(enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE))
                probs = torch.sigmoid(logits).cpu().numpy()[0]
                single_preds = (probs >= threshold).astype(int)
            single_missing = [LABEL_COLS[i] for i, v in enumerate(single_preds) if v == 0]
            st.session_state["single_missing"] = single_missing
            st.write("Eksikler:", single_missing if single_missing else "Yok")
    with col2:
        if st.button("LLM ile Öneri Üret") and single_req.strip():
            miss = st.session_state.get("single_missing", [])
            if not miss:
                st.warning("Önce eksikleri analiz edin veya eksik yok.")
            else:
                try:
                    if prompt_mode == "Strict":
                        prompt = build_strict_prompt(single_req, miss)
                    elif prompt_mode == "Zero-shot":
                        prompt = build_zeroshot_prompt(single_req, miss)
                    elif prompt_mode == "Few-shot":
                        prompt = build_fewshot_prompt(single_req, miss)
                    else:
                        prompt = build_llm_prompt(single_req, miss)
                    if active_llm is None:
                        st.error("LLM modeli seçin.")
                    else:
                        # Use optimized cached model loading
                        llm_model = load_llm_model(active_llm, llm_backend, llm_offline, use_cpu_only, max_memory_gb)
                        if llm_model is None:
                            st.error("LLM modeli yüklenemedi.")
                        else:
                            with st.spinner("LLM önerisi oluşturuluyor..."):
                                t0 = time.perf_counter()
                                
                                if llm_backend == "HF":
                                    # Optimize parameters based on model type
                                    if "gemma" in active_llm.lower():
                                        # Gemma models work better with different parameters
                                        out = llm_model(
                                            prompt, 
                                            max_new_tokens=48,  # Slightly more for Gemma
                                            do_sample=True, 
                                            num_beams=1,
                                            temperature=float(gen_temperature),
                                            top_p=float(gen_top_p),
                                            pad_token_id=llm_model.tokenizer.eos_token_id
                                        )
                                    else:
                                        # Standard parameters for other models
                                        out = llm_model(
                                            prompt, 
                                            max_new_tokens=32,  # Very small for memory efficiency
                                            do_sample=gen_temperature > 0.15, 
                                            num_beams=1,  # Single beam for memory
                                            temperature=float(gen_temperature),
                                            pad_token_id=llm_model.tokenizer.eos_token_id
                                        )
                                    gens = []
                                    for _ in range(int(max(1, k_candidates))):
                                        o = llm_model(
                                            prompt,
                                            max_new_tokens=64,
                                            min_new_tokens=24,
                                            do_sample=True,
                                            num_beams=1,
                                            temperature=float(gen_temperature),
                                            top_p=float(gen_top_p),
                                            no_repeat_ngram_size=2,
                                            repetition_penalty=1.15,
                                            pad_token_id=llm_model.tokenizer.eos_token_id
                                        )
                                        gens.append(o[0].get('generated_text','').strip())
                                    text = select_best_candidate(gens, single_req, miss)
                                    if strict_retry:
                                        r_base = _normalize_text(single_req)
                                        r_out = _normalize_text(text)
                                        if difflib.SequenceMatcher(None, r_base, r_out).ratio() >= float(sim_threshold):
                                            retry_prompt = (
                                                build_llm_prompt(single_req, miss)
                                                + "\n\nNot: Yukarıdaki cümleyi tekrar etme; yeni, daha NET, ÖLÇÜLEBİLİR ve DOĞRULANABİLİR tek cümle üret."
                                            )
                                            out2 = llm_model(
                                                retry_prompt, 
                                                max_new_tokens=48,
                                                do_sample=True,
                                                num_beams=1,
                                                temperature=float(gen_temperature),
                                                top_p=float(gen_top_p),
                                                pad_token_id=llm_model.tokenizer.eos_token_id
                                            )
                                            text = out2[0].get('generated_text','').strip()
                                else:  # GGUF
                                    gens = []
                                    for _ in range(int(max(1, k_candidates))):
                                        o = llm_model.create_completion(
                                            prompt=prompt, 
                                            max_tokens=64,
                                            temperature=float(gen_temperature), 
                                            top_p=float(gen_top_p),
                                            repeat_penalty=1.1
                                        )
                                        gens.append(o["choices"][0]["text"].strip())
                                    text = select_best_candidate(gens, single_req, miss)
                                    if strict_retry:
                                        r_base = _normalize_text(single_req)
                                        r_out = _normalize_text(text)
                                        if difflib.SequenceMatcher(None, r_base, r_out).ratio() >= float(sim_threshold):
                                            retry_prompt = (
                                                build_llm_prompt(single_req, miss)
                                                + "\n\nNot: Yukarıdaki cümleyi tekrar etme; yeni, daha NET, ÖLÇÜLEBİLİR ve DOĞRULANABİLİR tek cümle üret."
                                            )
                                            out2 = llm_model.create_completion(
                                                prompt=retry_prompt,
                                                max_tokens=48,
                                                temperature=float(gen_temperature),
                                                top_p=float(gen_top_p),
                                                repeat_penalty=1.1
                                            )
                                            text = out2["choices"][0]["text"].strip()
                                
                                dt = time.perf_counter() - t0
                                rb_text = rule_based_improvement(single_req, miss)
                                if force_llm_output:
                                    attempt = 0
                                    best_text = text
                                    while hard_constraints and attempt < 3:
                                        cleaned = _sanitize_llm_text(best_text)
                                        if validate_requirement_output(cleaned):
                                            break
                                        attempt += 1
                                        strict = build_strict_prompt(single_req, miss)
                                        o = llm_model(
                                            strict,
                                            max_new_tokens=64,
                                            min_new_tokens=24,
                                            do_sample=True,
                                            num_beams=1,
                                            temperature=float(gen_temperature),
                                            top_p=float(gen_top_p),
                                            no_repeat_ngram_size=2,
                                            repetition_penalty=1.15,
                                            pad_token_id=llm_model.tokenizer.eos_token_id
                                        )
                                        best_text = o[0].get('generated_text','').strip()
                                    cleaned = _sanitize_llm_text(best_text)
                                    if not validate_requirement_output(cleaned):
                                        cleaned = _sanitize_llm_text(select_best_candidate(gens, single_req, miss))
                                    final_text = f"İyileştirilmiş gereksinim: {cleaned}."
                                else:
                                    final_text = enforce_improvement(single_req, text, miss, similarity_threshold=float(sim_threshold))
                                if show_prompt:
                                    with st.expander("Gönderilen prompt"):
                                        st.code(prompt)
                                st.text_area("İyileştirilmiş Gereksinim", value=final_text, height=180)
                                if not force_llm_output and final_text.strip() == rb_text.strip():
                                    st.caption("Kaynak: LLM çıktısi normalleştirildi veya kural tabanlı metne yakın olduğu için kural tabanlı metin kullanıldı.")
                                else:
                                    st.caption("Kaynak: LLM (üretilen metin korundu)")
                                with st.expander("Ham LLM çıktısı"):
                                    st.write(text)
                                st.info(f"Süre: {dt:.2f} sn")
                except Exception as e:
                    st.error(f"LLM çalıştırılamadı: {e}")
                    # Fallback to rule-based improvement
                    final_text = rule_based_improvement(single_req, miss)
                    st.text_area("İyileştirilmiş Gereksinim (Kural tabanlı)", value=final_text, height=180)
                        
    # end uploaded path
