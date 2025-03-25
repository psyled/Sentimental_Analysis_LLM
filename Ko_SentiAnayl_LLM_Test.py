import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset
import torch


# 한국어 지원 T5 모델 사용
model_name = "KETI-AIR/ke-t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)



# ✅ 저장된 모델 로드 및 테스트
def generate_response(user_input):
    model = T5ForConditionalGeneration.from_pretrained("./trained_model_ko")
    tokenizer = T5Tokenizer.from_pretrained("./trained_model_ko")
    input_text = f"사용자: {user_input} \n AI:"
    
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    output_ids = model.generate(
        **inputs, 
        max_length=100,   # 응답 길이 조정
        min_length=10,    # 최소 길이 설정
        temperature=0.7,  # 생성 다양성 조절
        top_p=0.9,        # 확률 기반 샘플링
        num_return_sequences=1,
        do_sample=True
    )
    
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return response

# ✅ 샘플 테스트
sample_inputs = [
    "Why does work never seem to end? I'm so angry.",
    "It's better if I just handle it myself. I don't want to burden others.",
    "I've been feeling so depressed and exhausted lately."
]

for sample in sample_inputs:
    print(f"사용자: {sample}")
    print(f"AI 응답: {generate_response(sample)}")
    print("-")