from torch.utils.data import Dataset
import json


class ChatData(Dataset):
    def __init__(self, path:str, tokenizer):
        self.data = json.load(open(path, "r"))

        self.dataset = []

        for d in self.data["intents"]:
          for p in d["patterns"]:
            self.dataset.append(p)
            self.dataset.append(d["responses"][0])

        for id, i in enumerate(self.dataset):
            try:
                self.dataset[id] = "<startofstring> "+i+" <bot>: "+self.dataset[id+1]+" <endofstring>"
            except:
                break

        self.dataset_encoded = tokenizer(self.dataset, max_length=90, truncation=True, padding="max_length", return_tensors="pt")
        self.input_ids = self.dataset_encoded['input_ids']
        self.attention_mask = self.dataset_encoded['attention_mask']

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, id):
        return (self.input_ids[id], self.attention_mask[id])