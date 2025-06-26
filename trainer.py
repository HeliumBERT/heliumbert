from transformers import Trainer, TrainingArguments, AutoTokenizer
from model import HeliumbertConfig, HeliumbertForSequenceClassification
from datasets import load_dataset

# Load dataset
dataset = load_dataset("glue", "sst2")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")

def preprocess(examples):
    return tokenizer(examples["sentence"], truncation=True, padding="max_length", max_length=128)

dataset = dataset.map(preprocess, batched=True)
dataset = dataset.rename_column("label", "labels")
dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# Model config
config = HeliumbertConfig(num_labels=2, num_hidden_layers=6)
model = HeliumbertForSequenceClassification(config)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    weight_decay=0.01,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"].select(range(2000)),   # small subset for fast test
    eval_dataset=dataset["validation"].select(range(500)),
)

# Start training
trainer.train()
