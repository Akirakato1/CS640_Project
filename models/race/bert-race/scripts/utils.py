import re
from string import punctuation
from transformers import BertTokenizer

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