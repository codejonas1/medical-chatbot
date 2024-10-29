import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline

# pl_to_en = pipeline("translation", model="Helsinki-NLP/opus-mt-pl-en")

model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({"pad_token": "<pad>", 
                                "bos_token": "<startofstring>",
                                "eos_token": "<endofstring>"})
tokenizer.add_tokens(["<bot>:"])
model.resize_token_embeddings(len(tokenizer))

data = torch.load('model.pt', map_location=torch.device('cpu'), weights_only=True)

model.load_state_dict(data)
model.eval()

def convert(inp):
    # inp = pl_to_en(inp)[0]['translation_text']
    inp = "<startofstring> "+inp+" <bot>: "
    inp = tokenizer(inp, return_tensors="pt")
    input = inp["input_ids"]
    mask = inp["attention_mask"]
    output = model.generate(input, max_new_tokens=100, attention_mask=mask, do_sample=True)
    output = tokenizer.decode(output[0], skip_special_tokens=True)
    return output

while True:
    inp = input("You: ")
    if inp == 'q':
        print("See you!")
        break

    print(convert(inp))
    print('\n')
