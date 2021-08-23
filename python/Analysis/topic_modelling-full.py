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
# Plotting tools
#import pyLDAvis
#import pyLDAvis.gensim_models  # don't skip this

# Load data
dfs = []
#for file in glob.glob("selfdata/final/*.csv"):
for file in glob.glob("selfdata/final/vad/*.csv"):
    dfs.append(pd.read_csv(file))
df = pd.concat(dfs)
df.dropna(subset=['Adjectives'], inplace=True) # somekind of mistake
# literal evaluations

df = df[~df.subreddit.isna()]
df = df.loc[~df.subreddit.str.startswith("Q")] 

#df['Verbs'] = df['Verbs'].map(literal_eval)
#df['Adjectives'] = df['Adjectives'].map(literal_eval)
df['Nouns'] = df['Nouns'].map(literal_eval)

#df['Descriptors_parsed'] = df['Descriptors_parsed'].map(literal_eval)
#df['Verbs_parsed'] = df['Verbs_parsed'].map(literal_eval)

# Train on the comments that refer to MULTIPLE people
df = df.groupby('id').filter(lambda group: len(group) > 1)
df.drop_duplicates(subset='id', inplace=True)

class topic_model:
    def __init__(self, df, n_topics):
        self.stopwords = STOPWORDS
        self.text = df.map(lambda words: [word for word in words if word not in self.stopwords]) # remove stopwords
        #self.text = df.map(preprocess)
        self.dictionary = gensim.corpora.Dictionary(self.text)
        self.bow_corpus = [self.dictionary.doc2bow(doc) for doc in self.text]
        self.tfidf = models.TfidfModel(self.bow_corpus)
        self.corpus_tfidf = self.tfidf[self.bow_corpus]
        self.lda_model = gensim.models.LdaMulticore(self.corpus_tfidf, num_topics=n_topics, id2word=self.dictionary, passes=2, workers=4)
    
    def print_topics(self,filename):
        with open(filename,"w+") as f:
            for idx, topic in self.lda_model.print_topics(-1):
                f.write('Topic: {} Word: {} \n'.format(idx, topic))
    
#    def display_topic_distributions(self):
#        pyLDAvis.enable_notebook()
#        vis = pyLDAvis.gensim_models.prepare(self.lda_model, self.corpus_tfidf, self.dictionary)
#        return vis
        
    def model_coherence(self):
        print('Perplexity: ', self.lda_model.log_perplexity(self.corpus_tfidf))  # a measure of how good the model is. lower the better.
        # Compute Coherence Score
        self.coherence_model_lda = CoherenceModel(model=self.lda_model, texts=self.text, dictionary=self.dictionary, coherence='c_v')
        self.coherence_lda = self.coherence_model_lda.get_coherence()
        print('Coherence Score: ', self.coherence_lda)
    
    def test_examples(self,tests):
        other_corpus = [self.dictionary.doc2bow(t) for t in self.text[tests]]
        for i,doc in enumerate(other_corpus):
            print(self.raw_text[tests[i]])
            print(self.lda_model[doc])

    def save_model(self,filename):
        self.temp_file = datapath(filename)
        self.lda_model.save(self.temp_file)

num_topics = [5, 10, 15, 20, 25, 30, 40, 50]

for n in num_topics:

    subsample = df['Nouns'].sample(n=1000000)
    print(n, "------------------")

    print("\nNouns") 
    #Nouns, duh
    gen = topic_model(subsample,n_topics=n)
    gen.print_topics(f"nn-{n}.txt")
    gen.model_coherence()
    gen.save_model(f"nn-{n}")
    print("\n\n")
