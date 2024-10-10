import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({"pad_token": "<pad>", 
                                "bos_token": "<startofstring>",
                                "eos_token": "<endofstring>"})
tokenizer.add_tokens(["<bot>:"])
model.resize_token_embeddings(len(tokenizer))

data = torch.load('model.pt', map_location=torch.device('cpu'))

model.load_state_dict(data)
model.eval()

def saintize_data(txt):
    txt = txt.replace(" <bot>: ", "")

    return txt

def convert(inp):
    inp = "<startofstring> "+inp+" <bot>: "
    inp = tokenizer(inp, return_tensors="pt")
    input = inp["input_ids"]
    mask = inp["attention_mask"]
    output = model.generate(input, max_length=100, attention_mask=mask, do_sample=True)
    output = tokenizer.decode(output[0], skip_special_tokens=True)
    output = saintize_data(output)
    return output

while True:
    inp = input("You: ")
    if inp == 'q':
        print("See you!")
        break

    print(convert(inp))








