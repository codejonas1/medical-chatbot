from __future__ import annotations
import bentoml
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

@bentoml.service(
    http={
        "port": 5000,
        "cors": {
            "enabled": True,
            "access_control_allow_origins": ["*"],
            "access_control_allow_methods": ["GET", "OPTIONS", "POST", "HEAD", "PUT"],
            "access_control_allow_credentials": True,
            "access_control_allow_headers": ["*"],
            "access_control_max_age": 1200,
            "access_control_expose_headers": ["Content-Length"]
    }},
    resources={"cpu": "2"},
    tracing={"timeout": 5},
)
class Chat:
    def __init__(self) -> None:
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.add_special_tokens({"pad_token": "<pad>", 
                                        "bos_token": "<startofstring>",
                                        "eos_token": "<endofstring>"})
        self.tokenizer.add_tokens(["<bot>:"])
        self.model.resize_token_embeddings(len(self.tokenizer))

        data = torch.load('model.pt', map_location=torch.device('cpu'), weights_only=True)

        self.model.load_state_dict(data)
        self.model.eval()


    def convert(self, inp):
        inp = "<startofstring> "+inp+" <bot>: "
        inp = self.tokenizer(inp, return_tensors="pt")
        input = inp["input_ids"]
        mask = inp["attention_mask"]
        output = self.model.generate(input, max_new_tokens=100, attention_mask=mask, do_sample=True)
        output = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return output

    @bentoml.api
    def chat(self, query: str) -> str:

        if not query:
            return "Error"
        
        output = self.convert(query)
        return output