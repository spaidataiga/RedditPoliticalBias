"""
Maps Valence, Arousal, and Dominance values to each comment based on the VAD lexicon from NRC.
Used to calculate Sentimental Bias (via lexicon).
"""

import pandas as pd
import glob
import spacy
import numpy as np

nlp = spacy.load('en_core_web_lg')
vad = pd.read_csv("nrc_lexica/NRC-VAD-Lexicon.txt",index_col="Word", sep="\t")

def map_vad(text):
    doc = nlp(text)
    v = 0
    a = 0
    d = 0
    c = 0
    for token in doc:
        if token.text.lower() in vad.index:
            v += vad.loc[token.text.lower(),'Valence']
            a += vad.loc[token.text.lower(),'Arousal']
            d += vad.loc[token.text.lower(),'Dominance']
            c += 1
    if c > 0:
        v = v/c
        a = a/c
        d = d/c
    return v, a, d

for f in glob.glob("selfdata/final/parsed/*.csv"):
    filename = f[-11:]
    df = pd.read_csv(f)
    df['Valence'], df['Arousal'], df['Dominance'] = zip(*df['body'].map(map_vad))
    df.to_csv(f"selfdata/final/vad/{filename}",index=False)
