import re
import pandas as pd
import unicodedata
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub("\n", " ", s)
    s = re.sub(r"([.!?])", r" \1 ", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

df = pd.read_csv(f"valid_set_clean.tsv", sep="\t", names=['body', 'sex'])
df.body = df.body.map(normalizeString)
df.to_csv("model/valid_set_clean.tsv", sep="\t", header=False)

df = pd.read_csv(f"test_set_clean.tsv", sep="\t", names=['body', 'sex', 'loc'])
df.body = df.body.map(normalizeString)
df.to_csv("model/test_set_clean.tsv", sep="\t", header=False)
