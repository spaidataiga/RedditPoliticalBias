"""
This file generates the null models for each subset of the dataset that is used to calculate
the null distribution of the assortativity of the dataset (used to calculate the Combinatorial Bias)
"""

import pandas as pd
from ast import literal_eval
import glob
import numpy as np

def assortivity(G1, G2, multi):
    # assumes how many men/women are mentioned matters!
    #multi = df.groupby('id').filter(lambda group: len(group) > 1)
    g1_ids = multi[multi['sex'] == G1]['id']
    from_g1 = multi[multi['id'].isin(set(g1_ids))]
    P_g2_fromg1 = from_g1[from_g1['sex'] == G2].shape[0]/ len(g1_ids) ## should this be a set or not?
    return(P_g2_fromg1)

def cis_assortivity(G, multi):
    # Assumes the NUMBER of connections for each ID doesn't matter
    g_ids = set(multi[multi['sex'] == G]['id'])
    ## see where 'female' in value counts is > 2
    subset = multi.groupby(['id','sex'])['NEL'].count().unstack()
    P_fromgtog = (subset[subset[G] > 1.0][G].sum() - subset[subset[G] > 1.0][G].shape[0])/ len(g_ids)
    #P_g = subset[subset[G] > 0.0].shape[0]/ len(set(multi['id']))
    #print(P_fromgtog, P_g)
    return P_fromgtog

dfs = []
for f in glob.glob("selfdata/final/vad/201*.csv"): # Read from dataset here
    dfs.append(pd.read_csv(f))

# # Remove corruptions if present
# df = pd.concat(dfs).drop(columns=["Unnamed: 0"])
# df.dropna(subset=['Adjectives'], inplace=True) # somekind of mistake

# Only look at comments that mention more than one entity
multi = df.groupby('id').filter(lambda group: len(group) > 1)

# Subset the data
groups = ['left','right','alt_right']
subs = {'left': ['Liberal', 'SocialDemocracy', 'socialism', 'alltheleft', 'neoliberal', 'democrats'],
        'right':  ['Libertarian', 'Conservative', 'Republican'],
        'alt_right': ['The_Donald']}

for side in groups:
    subset = df[df.subreddit.isin(subs[side])]
    mm = open(f"null_models/{side}/male_male.txt", "w+") 
    mf = open(f"null_models/{side}/male_female.txt", "w+") 
    fm = open(f"null_models/{side}/female_male.txt", "w+") 
    ff = open(f"null_models/{side}/female_female.txt", "w+") 

    for i in range(10**4):
        shuffle = subset['id'].sample(frac=1).values
        new = subset[['id','NEL','sex']].copy()
        new['id'] = shuffle
        mf.write(str(assortivity('male','female', new)) + "\n")
        fm.write(str(assortivity('female','male', new)) + "\n")
        mm.write(str(cis_assortivity('male', new)) + "\n")
        ff.write(str(cis_assortivity('female', new)) + "\n")
    
    ff.close()
    mf.close()
    mm.close()
    fm.close()

