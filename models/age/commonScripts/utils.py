import re
from string import punctuation
from transformers import BertTokenizer
import torch

def clean(t):
#     # clean user name
#     t = re.sub(r"@\S*?\s*(:| |$)", "", t)
    # clean punctuations
    t=re.sub(r'[{}]+'.format(punctuation)," ",t)
    # clean line break
    t=re.sub(r'\\t|\\n|\n|\t'," ",t)
    # clean links
    URL_REGEX = re.compile(
        r"https?://[a-zA-Z0-9.?/&=:]*",
        re.IGNORECASE)
    t = re.sub(URL_REGEX, "", t)
    # merge spaces
    t = re.sub(r"\s+", " ", t)
    return t.strip()

def myTokenizer(data):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokens = tokenizer(data, padding=True, return_tensors='pt', truncation=True, max_length=512)
    input_ids = tokens["input_ids"].clone()
    attention_mask = tokens["attention_mask"]
    return input_ids, attention_mask


class BCEFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='elementwise_mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
 
    def forward(self, _input, target):
        pt = torch.sigmoid(_input)
        alpha = self.alpha
        loss = - alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - \
               (1 - alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss