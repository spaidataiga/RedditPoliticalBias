import pandas as pd
from ast import literal_eval
import glob
import numpy as np
import gensim
from gensim.parsing.preprocessing import STOPWORDS
from gensim import corpora, models
from gensim.test.utils import datapath
from gensim.models import CoherenceModel
import numpy as np
import logging

logging.basicConfig(filename='distribution.log', level=logging.DEBUG)
logging.debug("start logging")

logging.debug("Loading data")
# Need to make corpus with ENTIRE training dataset.
# Load data
dfs = []
#for file in glob.glob("selfdata/final/*.csv"):
for file in glob.glob("selfdata/final/vad/*.csv"):
    dfs.append(pd.read_csv(file))
df = pd.concat(dfs)
df.dropna(subset=['Adjectives'], inplace=True) # somekind of mistake
# literal evaluations
logging.debug("Loaded data. Processing")
df = df[~df.subreddit.isna()]
df = df.loc[~df.subreddit.str.startswith("Q")]
logging.debug("Looking exclusively at Nouns.")
subset = df[['id','Nouns']]
del df
#df['Verbs'] = df['Verbs'].map(literal_eval)
#df['Adjectives'] = df['Adjectives'].map(literal_eval)
logging.debug("Ensuring Nouns are read literally")
subset['Nouns'] = subset['Nouns'].map(literal_eval)
logging.debug("Finding only multiple-entity comments")
#df['Descriptors_parsed'] = df['Descriptors_parsed'].map(literal_eval)
#df['Verbs_parsed'] = df['Verbs_parsed'].map(literal_eval)

# Train on the comments that refer to MULTIPLE people
subset = subset.groupby('id').filter(lambda group: len(group) > 1)
subset.drop_duplicates(subset='id', inplace=True)

logging.debug("Isolated for multiple-person comments. Removing stopwords")
text = subset['Nouns'].map(lambda words: [word for word in words if word not in STOPWORDS]) # remove stopwords

del subset
logging.debug("Text processed. Creating dictionary")
 
dictionary = gensim.corpora.Dictionary(text)

logging.debug("Dictionary created. Loading model")

temp_file = datapath("nn-10")
model =  gensim.models.LdaMulticore.load(temp_file)

logging.debug("Model loaded. Loading test set")
test = pd.read_csv("selfdata/final/vad/2018-10.csv")
test.dropna(subset="Adjectives",inplace=True)
test = test[~test.subreddit.isna()]
test = test.loc[~test.subreddit.str.startswith("Q")]
test['Nouns'] = test['Nouns'].map(literal_eval)
test['Nouns'] = test['Nouns'].map(lambda words: [word for word in words if word not in STOPWORDS])

logging.debug("Analysing samples")
for point in df.sample(N=50):
    print(point.body)
    print(model[dictionary.doc2bow(point.Nouns)])
    print("")

