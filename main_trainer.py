import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import DataLoader
import tqdm
from torch.optim import Adam
import ChatData

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

def train(chatData, model, optim):

    epochs = 80

    for i in tqdm.tqdm(range(epochs)):
        for inputs, mask in chatData:
            inputs = inputs.to(device)
            mask = mask.to(device)
            optim.zero_grad()
            loss = model(inputs, attention_mask=mask, labels=inputs).loss
            loss.backward()
            optim.step()
            optim.zero_grad()

        print(convert("What to do if Cuts?"))
        torch.save(model.state_dict(), "model_state.pt")

def convert(inp):
    inp = "<startofstring> "+inp+" <bot>: "
    inp = tokenizer(inp, return_tensors="pt")
    input = inp["input_ids"].to(device)
    mask = inp["attention_mask"].to(device)
    output = model.generate(input, max_new_tokens=40, attention_mask=mask)
    output = tokenizer.decode(output[0])
    return output

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

tokenizer.add_special_tokens({"pad_token": "<pad>", 
                                "bos_token": "<startofstring>",
                                "eos_token": "<endofstring>"})
tokenizer.add_tokens(["<bot>:"])
model.resize_token_embeddings(len(tokenizer))

model = model.to(device)
model.train()
optim = Adam(model.parameters(), lr=1e-3)

dataset = ChatData("intents.json", tokenizer)
dataset =  DataLoader(dataset, batch_size=64)
train(dataset, model, optim)

print(convert("What to do if Cuts?"))