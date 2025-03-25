import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset
import torch

# 엑셀 파일 경로
excel_file_path = 'semtimental_Training2.xlsx'

# 엑셀 데이터 로드
df = pd.read_excel(excel_file_path)

# 대화 데이터 전처리
def prepare_data(df):
    data = []
    for _, row in df.iterrows():
        user_input = f"User: {row['사람문장1']} \n User: {row['사람문장2']} \n User: {row['사람문장3']}"
        system_response = f"System: {row['시스템문장1']} \n System: {row['시스템문장2']} \n System: {row['시스템문장3']}"
        data.append({"input_text": user_input, "target_text": system_response})
    return data

# 데이터셋 생성
data = prepare_data(df)
dataset = Dataset.from_pandas(pd.DataFrame(data))

# 모델 및 토크나이저 불러오기
model_name = "KETI-AIR/ke-t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# 데이터 토큰화
def tokenize_function(example):
    model_input = tokenizer(example["input_text"], padding="max_length", truncation=True, max_length=512)
    label = tokenizer(example["target_text"], padding="max_length", truncation=True, max_length=512)
    model_input["labels"] = label["input_ids"]
    return model_input

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 학습 설정
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# 모델 학습
trainer.train()

# 학습된 모델 저장
model.save_pretrained("./trained_model_ko")
tokenizer.save_pretrained("./trained_model_ko")
