from transformers import GPT2LMHeadModel, GPT2Tokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_dataset

def load_and_tokenize_dataset(file_path, tokenizer, block_size=128):
    dataset = load_dataset('text', data_files=file_path, split='train')

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=block_size)

    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    return tokenized_datasets

model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(model_name)

dataset_path = "cleaned_jokes.txt"
train_dataset = load_and_tokenize_dataset(dataset_path, tokenizer)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

training_args = TrainingArguments(
    output_dir="./gpt2-finetuned-joke",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

trainer.train()


model.save_pretrained("./gpt2-finetuned-joke")
tokenizer.save_pretrained("./gpt2-finetuned-joke")
