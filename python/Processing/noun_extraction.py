import spacy
import pandas as pd
import glob

nlp = spacy.load("en_core_web_lg")

def spacy_extract_words(text):
    doc = nlp(text)
    nouns = []
    for token in doc:
        if "\\n" in token.text: #remove this from formatting
            continue
        elif token.pos_ == 'NOUN':
            nouns.append(token.lemma_)
    return nouns

for f in glob.glob("selfdata/final/fixed/*"):
    df = pd.read_csv(f)
    file_name = f[21:]
    df['Nouns'] = df.body.apply(spacy_extract_words)

    # save final file.
    df.to_csv(f"selfdata/final/{file_name}",index=False)
