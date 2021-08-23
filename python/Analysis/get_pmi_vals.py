import pandas as pd
from ast import literal_eval
import glob
import numpy as np
import re, string

DELTA = 0.7

def return_PMI(row):
    fem_PMI = np.log( (row.female) / (P_fem * ((row.female + row.male)/N)))
    mal_PMI = np.log( (row.male) / (P_mal * ((row.female + row.male)/N)))
    return fem_PMI, mal_PMI

def return_PMId(row):
    fem_PMI = np.log( (row.female) / (Pd_fem * ((row.female + row.male)/D)))
    mal_PMI = np.log( (row.male) / (Pd_mal * ((row.female + row.male)/D)))
    return fem_PMI, mal_PMI

def return_cPMId(row):
    fem_cPMId = np.log( row.female / (Pd_fem * ((row.female + row.male)/D) + np.sqrt(Pd_fem) * np.sqrt(np.log(DELTA)/-2)))
    mal_cPMId = np.log( row.male / (Pd_mal * ((row.female + row.male)/D) + np.sqrt(Pd_mal) * np.sqrt(np.log(DELTA)/-2)))
    return fem_cPMId, mal_cPMId

dfs = []
for file in glob.glob("selfdata/final/vad/201*.csv"):
    dfs.append(pd.read_csv(file))
df = pd.concat(dfs)
del dfs
df.dropna(subset=['Descriptors_parsed'], inplace=True) # somekind of mistake
df.drop(columns=["Unnamed: 0"], inplace=True)
df = df[~(df.subreddit.str.startswith("Q")) | ~(df.subreddit.str.startswith("["))]
df = df[df['sex'].isin(['male','female'])]

df['Descriptors_parsed'] = df['Descriptors_parsed'].map(literal_eval)
df['Descriptors_parsed'] = df['Descriptors_parsed'].map(set)

left = ['Liberal', 'SocialDemocracy', 'socialism', 'alltheleft', 'neoliberal', 'democrats']
right = ['Libertarian', 'Conservative', 'Republican']
alt_right = ['The_Donald']
euro = ['ireland', 'unitedkingdom', 'europe']
non_euro = ['canada', 'uspolitics', 'newzealand', 'australia', 'india']
eng = ['canada', 'uspolitics', 'newzealand', 'australia', 'india', 'ireland', 'unitedkingdom']
non_eng = ['europe']

fem_freq = {}
mal_freq = {}
for doc in df[df['sex'] == 'female'].Descriptors_parsed:
    for adj in doc: #ensure the same word isn't counted twice
        if adj.lower() in fem_freq.keys():
            fem_freq[adj.lower()] += 1
        else:
            fem_freq[adj.lower()] = 1
            mal_freq[adj.lower()] = 0

for doc in df[df['sex'] == 'male'].Descriptors_parsed:
    for adj in doc: #ensure the same word isn't counted twice
        if adj.lower() in mal_freq.keys():
            mal_freq[adj.lower() ] += 1
        else:
            mal_freq[adj.lower() ] = 1
            fem_freq[adj.lower() ] = 0

array = pd.DataFrame.from_dict([fem_freq,mal_freq])
array.index = ['female', 'male']

array_T = array.transpose()
array_T = array_T[(array_T.female > 2) & (array_T.male > 2)] # only look at words that appear at least 3 times in both sample sets.

W_adj = sum(df.Descriptors_parsed.map(len))

N =  df.shape[0]
P_fem = df[df['sex'] == 'female'].shape[0]
P_mal = df[df['sex'] == 'male'].shape[0]

array_T['PMI_female'], array_T['PMI_male'] = zip(*array_T.apply(lambda row: return_PMI(row), axis=1))

df['Descriptors_parsed'] = df['Descriptors_parsed'].map(list)

entity_level = df.groupby('NEL').agg({'Descriptors_parsed': list})
entity_level['Descriptors_parsed'] = entity_level['Descriptors_parsed'].apply(lambda x: [item for sublist in x for item in sublist])
entity_level['Descriptors_parsed'] = entity_level['Descriptors_parsed'].map(set)

mapper = df[['NEL','sex']].drop_duplicates('NEL').set_index('NEL').to_dict()['sex']
entity_level['sex'] = entity_level.index.map(mapper)

punct = string.punctuation
punct = punct.replace('-','')
punct = punct + "\n\t"

pattern = re.compile("\w*[" + re.escape(punct) + "]+\w*")

d_fem_freq = {}
d_mal_freq = {}
for ent in entity_level[entity_level['sex'] == 'female'].Descriptors_parsed:
    for adj in ent: #ensure the same word isn't counted twice
        if adj.lower() in d_fem_freq.keys():
            d_fem_freq[adj.lower()] += 1
        else:
            if not pattern.match(adj):
                d_fem_freq[adj.lower()] = 1
                d_mal_freq[adj.lower()] = 0

for ent in entity_level[entity_level['sex'] == 'male'].Descriptors_parsed:
    for adj in ent: #ensure the same word isn't counted twice
        if adj.lower() in d_mal_freq.keys():
            d_mal_freq[adj.lower() ] += 1
        else:
            if not pattern.match(adj):
                d_fem_freq[adj.lower()] = 1
                d_mal_freq[adj.lower()] = 0

D = entity_level.shape[0]
Pd_fem = entity_level[entity_level['sex'] == 'female'].shape[0]
Pd_mal = entity_level[entity_level['sex'] == 'male'].shape[0]

d_array = pd.DataFrame.from_dict([d_fem_freq,d_mal_freq])
d_array.index = ['female', 'male']
d_array_T = d_array.transpose()
d_array_T = d_array_T[(d_array_T.female > 2) & (d_array_T.male > 2)] # only look at words that appear at least 3 times in both sample sets.

d_array_T['PMI_female'], d_array_T['PMI_male'] = zip(*d_array_T.apply(lambda row: return_PMI(row), axis=1))

d_array_T['cPMId_female'], d_array_T['cPMId_male'] = zip(*d_array_T.apply(lambda row: return_cPMId(row), axis=1))

d_array_T['PMId_female'], d_array_T['PMId_male'] = zip(*d_array_T.apply(lambda row: return_cPMId(row), axis=1))

most_fem = d_array_T.sort_values(by=['PMI_female'], axis=0, ascending=False).iloc[:100]
least_fem = d_array_T.sort_values(by=['PMI_male'], axis=0, ascending=False).iloc[:100]

most_fem_d = d_array_T.sort_values(by=['PMId_female'], axis=0, ascending=False).iloc[:100]
least_fem_d = d_array_T.sort_values(by=['PMId_male'], axis=0, ascending=False).iloc[:100]

most_fem_c = d_array_T.sort_values(by=['cPMId_female'], axis=0, ascending=False).iloc[:100]
least_fem_c = d_array_T.sort_values(by=['cPMId_male'], axis=0, ascending=False).iloc[:100]

most_female = array_T.sort_values(by=['PMI_female'], axis=0, ascending=False).iloc[:100]
least_female = array_T.sort_values(by=['PMI_male'], axis=0, ascending=False).iloc[:100]

print("Number posts:", N)
print("Percentage posts fem", P_fem)
print("Percentage posts mal", P_mal)
print()
print("Number docs", D)
print("Percentage entities fem", Pd_fem)
print("Percentage entities mal", Pd_mal)

print("constant with delta", np.sqrt(Pd_fem) * np.sqrt(np.log(DELTA)/-2))
