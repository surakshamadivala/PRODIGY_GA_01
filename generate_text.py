from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import re


model_name = './gpt2-finetuned-joke'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)


input_text = " Why do bees have sticky hair? "
input_ids = tokenizer.encode(input_text, return_tensors='pt')


attention_mask = torch.ones(input_ids.shape, dtype=torch.long)


output = model.generate(
    input_ids,
    attention_mask=attention_mask,
    max_length=50,  
    num_return_sequences=1,  
    no_repeat_ngram_size=2,  
    temperature=0.7,  
    top_p=0.9,  
    top_k=50,  
    do_sample=True,  
    pad_token_id=tokenizer.eos_token_id  
)


generated_text = tokenizer.decode(output[0], skip_special_tokens=True)


def post_process_text(text):
    
    text = text.strip()  
    text = re.sub(r'\s+', ' ', text)  
    text = re.sub(r'\.\s*$', '.', text)  
    
    
    if '.' in text:
        text = text.split('.')[0] + '.'
    
    return text


processed_text = post_process_text(generated_text)


print(processed_text)
