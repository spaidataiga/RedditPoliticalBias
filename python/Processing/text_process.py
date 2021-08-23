import sys

month = sys.argv[1]
year = sys.argv[2]

import pandas as pd
import unicodedata
import re
import numpy as np

df = pd.read_csv(f"selfdata/final/vad/{year}-{month}.csv")
df.drop_duplicates(subset='id',keep=False, inplace=True)

# Look at only comments about one entity

new = df[['body', 'sex', 'NEL', 'subreddit']]
del df

subs = pd.read_csv("sex_subwords.txt",sep=":",names=['word','translation'],index_col=0)['translation'].to_dict()

class Lang:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        if len(sentence.split(' ')) < 4:
            return
        for word in sentence.split(' '):
            self.addWord(word)
            
    def addSentenceNew(self,lst):
        if len(lst) < 4:
            return
        for word in lst:
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub("\n", " ", s)
    s = re.sub(r"([.!?])", r" \1 ", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

new['text'] = new.body.map(normalizeString)

new['processed'] = np.nan
for i in range(new.shape[0]):
    words = []
    for word in new.text.iloc[i].split(' '):
        # Replace OBVIOUSLY GENDERED words
        if word in subs.keys():
            words.append(subs[word])
        else:
            words.append(word)
    new['processed'].iloc[i] = words

# Remove comments with <4 words
new['len'] = new.processed.map(len)
new2 = new[new['len'] > 4]
del new

def createNewDic(df):
    input = Lang()
    print("Counting words...")
    for dp in df.processed:
        input.addSentenceNew(dp)
    print("Counted words:")
    print(input.n_words)
    return input
lang_new = createNewDic(new2)

# Only keep words that appear more than 50 times in dataset.
new2['processed_sub100000'] = ''
new2['processed_subsmaller'] = ''
for i in range(new2.shape[0]):
    words = []
    words_2 = []
    for word in new2.processed.iloc[i]:
        if lang_new.word2count[word] < 50:
            words.append('unk')
            words_2.append('unk')
        elif lang_new.word2count[word] < 100:
            words_2.append('unk')
            words.append(word)
        else:
            words.append(word)
            words_2.append(word)
    new2['processed_sub100000'].iloc[i] = words
    new2['processed_subsmaller'].iloc[i] = words_2

new2.to_csv(f"selfdata/final/train/{year}-{month}.csv")
