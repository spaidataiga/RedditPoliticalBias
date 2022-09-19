import logging
import spacy
import pandas as pd
import time
import numpy as np
import re
import datetime
from dateutil.relativedelta import relativedelta
import sys

month = sys.argv[1]
year = sys.argv[2]
split = sys.argv[3]

def map_keypath(path, refs):
    """ Follows the dictionary path from child to top parent comment. Returns list as a series of strings.
    Feed in the first value as a one element list. Also give the dictioanry of child to parent ID. """
    if path[0] in refs.keys():
        return map_keypath([refs[path[0]]] + path, refs)
    else:
        return '\n'.join(path)

date=datetime.date(int(year),int(month),1)
date_str = date.strftime('%Y-%m')
logging.basicConfig(filename=f'logs/step2/{date_str}/{split}.log', level=logging.DEBUG)
logging.debug(spacy.__version__)
logging.debug(date_str)
t00 = time.clock()
nlp = spacy.load('en_core_web_md')
logging.debug("Spacy loaded.")

# Load data
# Load data
df = pd.read_csv(f'selfdata/processed/splits/{date_str}/{date_str}_{split}.csv')
df.set_index('id',inplace=True)
cntxt = pd.read_csv(f'selfdata/comments/{date_str}_data.csv', index_col='id')
#Now add in non-archived comments (aka from posts from the last 6 months)
for i in range(1,7):
        new_day = date + relativedelta(months=-i)
        cntxt = pd.concat((cntxt,pd.read_csv(f"selfdata/comments/{new_day.strftime('%Y-%m')}_data.csv",index_col='id')))

logging.debug("All comments loaded. " + str(df.shape[0]))

logging.debug("Keeping only comments with context!")
# ensure that the to_id, if it's to a post, is addressed to a comment I have in the database!
ids = cntxt.index.unique().tolist()
logging.debug(str(len(ids)))
# drop all comments we do not have the parent/context for. In a while loop so that we always have context
while df[df["to_type"] == "t1"].query("to_id not in @ids").shape[0] > 0:
    logging.debug("DROP: " + str(df[df["to_type"] == "t1"].query("to_id not in @ids").shape[0]))
    df.drop(index=df[df["to_type"] == "t1"].query("to_id not in @ids").index, inplace=True)
    cntxt.drop(index=df[df["to_type"] == "t1"].query("to_id not in @ids").index, inplace=True)
    ids = cntxt.index.unique().tolist()
del ids
logging.debug("Adding context")

# Add comment context to each comment
references = df[df['to_type']=='t1']['to_id'].to_dict() # Dictionary of child to parent

# Create a path of reply-IDs for all comments that reply to comments
df['cmt_context'] = ''
df['cmt_context'].loc[df[df['to_type'] == 't1'].index] = df[df['to_type'] == 't1']['to_id'].map(lambda x: [x]).apply(map_keypath, args=(references,))

# Keep only comments with named entities
logging.debug("Keeping only named entities!")
i = 0
logging.debug("OLD SIZE " + str(df.shape[0]))
t0= time.clock()
while i < df.shape[0]:
    if df.cmt_context.iloc[i] == '':
        i += 1
        continue
    context = df.cmt_context.iloc[i].split('\n')
    context = [x for x in context if x in df.index]

    if len(context) > 0:
        if not df.loc[context]['named'].any():
            index = df.index[i]
            df.drop(context, inplace=True)
            df.drop(index, inplace=True)
            continue
    del context
    if i%10000 == 0:
        logging.debug(i)
    i += 1
    
logging.debug("NEW SIZE " + str(df.shape[0]))
t1 = time.clock() - t0
logging.debug("Time elapsed: " + str(t1)) # CPU seconds elapsed (floating point)

logging.debug("Saving")
# Save the data for NEL
df.to_csv(f"selfdata/processed2/{date_str}/{date_str}_{split}.csv")
logging.debug("Done.")
