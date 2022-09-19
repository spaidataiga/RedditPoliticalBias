import pandas as pd
import sys
import glob
import datetime
from ast import literal_eval
import spacy
import numpy as np

month = sys.argv[1]
year = sys.argv[2]
date=datetime.date(int(year),int(month),1)
date_str = date.strftime('%Y-%m')

# Merge splits into one file
dfs = []
for f in glob.glob(f"selfdata/processed4/{date_str}/*.csv"):
    dfs.append(pd.read_csv(f))
df = pd.concat(dfs) # merge into one file

# I accidentally dropped the column ID in my earlier processing, so I need to make one to link identical comments together
#df['id']=df.groupby(['created_at','to_id']).ngroup().add(1) # group by created_at and to_id to do so

drop_columns = ['link_id','created_at','to_id','post_named','cmt_named','named']
df.drop(columns=drop_columns, inplace=True) # drop unnecessary columns

#remove linebreaks that I didn't early
df['body'] = df['body'].str.replace("\\n",'')

# Make sure we can read list as a true list
df['Names'] = df['Names'].apply(literal_eval)

# Convert wikidata info into dictionaries for mapping
import json
with open('politician_data_final.json') as f:
    wikidata = json.load(f)

genders = {}
for entry in wikidata:
    genders[entry['id']] = entry['sex_or_gender']
    
ethnicities = {}
for entry in wikidata:
    if "ethnicity" in entry.keys():
        ethnicities[entry['id']] = entry['ethnicity']
        
parties = {}
for entry in wikidata:
    if "party" in entry.keys():
        parties[entry['id']] = entry['party']

origins = {}
for entry in wikidata:
    if "origin" in entry.keys():
        origins[entry['id']] = entry['origin']
        
ages = {}
for entry in wikidata:
    if "DOB" in entry.keys():
        ages[entry['id']] = entry['DOB']
        
positions = {}
for entry in wikidata:
    if "positionheld" in entry.keys():
        positions[entry['id']] = entry['positionheld']

        
## This may be inaccurate, check.
first_names = {}
last_names = {}
for entry in wikidata:
    if "givenname" in entry.keys():
        first_names[entry['id']] = entry['givenname']
    else:
        first_names[entry['id']] = entry['name'].split(' ')[0]
    if "familyname" in entry.keys():
        last_names[entry['id']] = entry['familyname']
    else:
        last_names[entry['id']] = entry['name'].split(' ')[-1]

# MAP WIKIDATA INFO IF AVAILABLE

df['sex'] = df['NEL'].map(genders)
df['sex'] = df['sex'].str.replace('cisgender ', '') #most are cisgender, new addition so remove the clarification to ensure consistency.
df['ethnicity'] = df['NEL'].map(ethnicities)
df['origin'] = df['NEL'].map(origins)
df['DOB'] = df['NEL'].map(ages)
df['highest_position'] = df['NEL'].map(positions)
df['party'] = df['NEL'].map(parties)

# Assess name used -- MAY NEED TO DEBUG
df['entity_given_name'] = df['NEL'].map(first_names)
df['entity_family_name'] = df['NEL'].map(last_names)
df['given_name_used'] = df.apply(lambda x: x['entity_given_name'] in x['Names'], axis=1)
df['family_name_used'] = df.apply(lambda x: x['entity_family_name'] in x['Names'], axis=1)

df['New_Names'] = df['Names'].map(lambda items: [x.split(' ') for x in items][0])
df['full_name_used'] = df.apply(lambda x: x['entity_given_name'] in x['New_Names'], axis=1) & df.apply(lambda x: x['entity_family_name'] in x['New_Names'], axis=1)
df.drop(columns=['New_Names'],inplace=True)
df['nickname_used'] = ~((df['given_name_used'] | df['family_name_used']) | df['full_name_used'])
df.dropna(axis=1,how='all',inplace=True)

#Extract verbs and adjectives while we're at it

nlp = spacy.load("en_core_web_lg")
vad = pd.read_csv("nrc_lexica/NRC-VAD-Lexicon.txt",index_col="Word", sep="\t")

def spacy_extract_words(text):
    doc = nlp(text)
    adjs = []
    verbs = []
    nouns = []
    for token in doc:
        if "\\n" in token.text: #remove this from formatting
            continue
        if token.pos_ == 'ADJ':
            adjs.append(token.lemma_)
        elif token.pos_ == 'VERB':
            verbs.append(token.lemma_)
        elif token.pos_ == 'NOUN':
            nouns.append(token.lemma_)
    return adjs, verbs, nouns

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


def map_vad(text):
    doc = nlp(text)
    v = 0
    a = 0
    d = 0
    for token in doc:
        if token.text.lower() in vad.index:
            v += vad.loc[token.text.lower(),'Valence']
            a += vad.loc[token.text.lower(),'Arousal']
            d += vad.loc[token.text.lower(),'Dominance']
    v = v/len(doc)
    a = a/len(doc)
    d = d/len(doc)
    return v, a, d

remove = ['Denmark', 'Polska', 'norge']

df = df[~df.subreddit.isin(remove)]

drop = [item for item in df.columns if 'Unnamed' in item]
df.drop(columns=drop, inplace=True) #Drop any unnamed columns.

df['Adjectives'], df['Verbs'], df['Nouns'] = zip(*df.body.apply(spacy_extract_words))
df['Descriptors_parsed'], df['Verbs_parsed'], df['Relation'] = zip(*df['body'].map(dep_parse))

df['Valence'], df['Arousal'], df['Dominance'] = zip(*df['body'].map(map_vad))

# save final file.
df.to_csv(f"selfdata/final/vad/{date_str}.csv",line_terminator='\r',index=False)
