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
df['Descriptors_parsed'] = df['Descriptors_parsed'].map(set).map(list)

left = ['Liberal', 'SocialDemocracy', 'socialism', 'alltheleft', 'neoliberal', 'democrats']
right = ['Libertarian', 'Conservative', 'Republican']
alt_right = ['The_Donald']

sides = [left, right, alt_right]
side_names = ["left", "right", "alt_right"]
punct = string.punctuation
punct = punct.replace('-','')
punct = punct + "\n\t"

pattern = re.compile("\w*[" + re.escape(punct) + "]+\w*")

for i in range(len(sides)):
    side = sides[i]
    subset = df[df.subreddit.isin(side)]
    entity_level = subset.groupby('NEL').agg({'Descriptors_parsed': list})
    entity_level['Descriptors_parsed'] = entity_level['Descriptors_parsed'].apply(lambda x: [item for sublist in x for item in sublist])
    entity_level['Descriptors_parsed'] = entity_level['Descriptors_parsed'].map(set)

    mapper = subset[['NEL','sex']].drop_duplicates('NEL').set_index('NEL').to_dict()['sex']
    entity_level['sex'] = entity_level.index.map(mapper)
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

    d_array_T['PMI_female'], d_array_T['PMI_male'] = zip(*d_array_T.apply(lambda row: return_PMId(row), axis=1))

    most_fem_c = d_array_T.sort_values(by=['PMI_female'], axis=0, ascending=False).iloc[:100]
    least_fem_c = d_array_T.sort_values(by=['PMI_male'], axis=0, ascending=False).iloc[:100]

    most_fem_c.to_csv(f"PMI/most_female_{side_names[i]}.csv")
    least_fem_c.to_csv(f"PMI/most_male_{side_names[i]}.csv")
