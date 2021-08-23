import pandas as pd
import spacy
import glob

nlp = spacy.load('en_core_web_lg')

def dep_parse(txt):
    adj = []
    vb = []
    positions = []
    doc = nlp(txt)
    for token in doc:
        if token.text == 'NAME':
            positions.append(token.dep_)
            for kid in token.children:
                if kid.pos_ in ['ADJ', 'ADV', 'NOUN']:
                    adj.append(kid.lemma_)
    #         predicates_elems[token.i] = token
    #         root_index = token.i
        if 'NAME' in [x.text for x in list(token.children)]:
            if token.pos_ in ['VERB','AUX']:
                vb.append(token.lemma_)
            for child in token.children:
                if child.pos_ in ['ADJ', 'ADV', 'NOUN']:
                    if child.text != 'NAME':
                        adj.append(child.lemma_)
                        for kid in child.children:
                            if kid.pos_ in ['ADJ', 'ADV', 'NOUN']:
                                adj.append(kid.lemma_)
    return adj, vb, positions

for f in glob.glob("selfdata/final/*.csv"):
    df = pd.read_csv(f)
    filename = f[-11:]
    df['Descriptors_parsed'], df['Verbs_parsed'], df['Relation'] = zip(*df['body'].map(dep_parse))
    df.to_csv(f"selfdata/final/parsed/{filename}", index=False)
