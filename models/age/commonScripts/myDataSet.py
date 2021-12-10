from torch.utils.data import Dataset
import utils
from transformers import BertTokenizer, BertModel
import torch


class MyDataset(Dataset):
    def __init__(self, df):
        self.data = list(map(utils.clean, df["tweets"].tolist()))
        self.label = list(map(int, (df["label"]).tolist()))

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]
        return data, label

    def __len__(self):
        return len(self.label)

    def getLabels(self):
        return self.label


class MyTokenizerDataset(Dataset):
    def __init__(self, df):
        datas = list(map(utils.clean, df["tweets"].tolist()))
        self.input_ids = []
        self.attention_masks = []
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        tokens = tokenizer(datas, padding=True, return_tensors='pt', truncation=True, max_length=512)
        self.input_ids = tokens["input_ids"]
        self.attention_masks = tokens["attention_mask"]
        self.label = list(map(int, (df["label"]).tolist()))

    def __getitem__(self, index):
        #         data = self.data[index]
        input_ids = self.input_ids[index]
        attention_mask = self.attention_masks[index]
        label = self.label[index]
        return input_ids, attention_mask, label

    def __len__(self):
        return len(self.label)

    def getLabels(self):
        return self.label


class MyBertDataset(Dataset):
    def __init__(self, df):
        datas = list(map(utils.clean, df["tweets"].tolist()))
        self.input_ids = []
        self.attention_masks = []
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert = BertModel.from_pretrained("bert-base-uncased").cuda()
        tokens = tokenizer(datas, padding=True, return_tensors='pt', truncation=True, max_length=512)
        with torch.no_grad():
            features = bert(tokens['input_ids'].cuda(), attention_mask=tokens['attention_mask'].cuda())[0][:, 0]

        #         self.input_ids=tokens["input_ids"]
        #         self.attention_masks=tokens["attention_mask"]
        self.features = features

        self.label = list(map(int, (df["label"]).tolist()))

    def __getitem__(self, index):
        #         data = self.data[index]
        input_ids = self.input_ids[index]
        #         attention_mask=self.attention_masks[index]
        #         label = self.label[index]
        feature = self.features[index]
        #         return input_ids,attention_mask, label
        return feature, label

    def __len__(self):
        return len(self.label)

    def getLabels(self):
        return self.label
