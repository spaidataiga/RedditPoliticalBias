import logging
import spacy
import neuralcoref
import pandas as pd
import os
import time
import numpy as np
import re
import datetime
from dateutil.relativedelta import relativedelta
import sys

month = sys.argv[1]
year = sys.argv[2]
split = sys.argv[3]

def resolve(context, body):
    if type(body) == float:
        return np.nan
    if type(context) == float: # context might be NaN
        text = body
    else:
        text = context + "\t" + body
    logging.debug(text) ## JUST TO DEBUG 0631
    doc1 = nlp(text)
    wcontext = doc1._.coref_resolved.rsplit('\t', 1)
    return wcontext[1] if len(wcontext) > 1 else wcontext #### Sometimes the split doesn't work

date=datetime.date(int(year),int(month),1)
date_str = date.strftime('%Y-%m')
if not os.path.exists(f"selfdata/processed3/{date_str}"):
    os.makedirs(f"selfdata/processed3/{date_str}")
    os.makedirs(f"logs/step3/{date_str}")
logging.basicConfig(filename=f'logs/step3/{date_str}/{split}.log', level=logging.DEBUG)
logging.debug(spacy.__version__)
logging.debug(date_str)
t00 = time.clock()
nlp = spacy.load('en_core_web_md')
neuralcoref.add_to_pipe(nlp)
logging.debug("Spacy and Neural Coref loaded.")

# Load data
df = pd.read_csv(f'selfdata/processed2/{date_str}/{date_str}_{split}.csv')
df = df[df.id != 'eckia6l'] ## REMOVE TROUBLESOME COMMENT
df.set_index('id',inplace=True)
logging.debug("Total number of comments: " + str(df.shape[0]))

# Find IDs of parent comments
TO_DROP = ['f79px53','f34z0ew','f342d09','f32sfx4','f33imif','f221kvi', 'f232eq2' ,'eztkk4g','ezt16z0','ezq8x8x','ezorm9b','ezq1zp2','ezq3nbi','f17vxhx','f17uk0s','f17v2ns','f17vdzk','f17vhhg', 'eckfsfh', 'eckfnuz', 'eckf5ah', 'eckemv3', 'eckdqdq','eckcuur', 'eckbrdl','eckbdz7']
responded = df['cmt_context'].loc[df['cmt_context'].notna()].map(lambda s: s.split('\n'))
tos = []
for value in responded.values:
    tos.extend(value)

### REMOVE MORE TROUBLESOME STUFF
tos = set(tos)
logging.debug("TO THIS MANY " + str(len(tos)))
tos.difference_update(TO_DROP)
logging.debug("AFTER REMOVAL " + str(len(tos)))

cntxt = pd.read_csv(f'selfdata/comments/{date_str}_data.csv', index_col='id')
cmt_reference = cntxt.loc[cntxt.index.isin(tos)]['body'].to_dict()
del cntxt
#Now add in non-archived comments (aka from posts from the last 6 months)
for i in range(1,7):
            if len(cmt_reference) == len(tos):
                break # if we have gotten all the comments referenced, we don't need to keep searching!
            new_day = date + relativedelta(months=-i)
            cntxt = pd.read_csv(f"selfdata/comments/{new_day.strftime('%Y-%m')}_data.csv",index_col='id')
            cmt_reference.update(cntxt.loc[cntxt.index.isin(tos)]['body'].to_dict())
            logging.debug(len(cmt_reference)/len(tos))
            del cntxt
logging.debug("Converting context to text.")
# Convert these IDs to the actual text
pattern = '|'.join(sorted(re.escape(k) for k in cmt_reference))
df['cmt_context'].fillna('',inplace=True)
df['cmt_context'] = df['cmt_context'].str.replace(pattern, lambda m: cmt_reference.get(m.group(0)), regex=True)

# Merge with post title and drop the not needed column
df['context'] = df['context'] + "\n" + df['cmt_context']
df.drop(columns='cmt_context', inplace=True)
del cmt_reference
del pattern

# Coreference resolution
logging.debug("Resolving coreferences")
t0= time.clock()
logging.debug(df.shape)
df['body'] = df.apply(lambda row: resolve(context = row['context'], body = row['body']), axis=1)
t1 = time.clock() - t0
logging.debug("Time elapsed: " + str(t1)) # CPU seconds elapsed (floating point)

# Remove context column (huge)
df.drop(columns=['context'], inplace=True)

logging.debug("Saving")
# Save the data for NEL

df.to_csv(f"selfdata/processed3/{date_str}/{date_str}_{split}.csv")
logging.debug("Done.")
logging.debug("Total time elapsed: " + str(time.clock() - t00))
