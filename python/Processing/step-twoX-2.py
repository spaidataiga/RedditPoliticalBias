import pandas as pd
import datetime
import os
import time
import numpy as np
import re
import json
import sys
import requests
import logging

from REL.mention_detection import MentionDetection
from REL.utils import process_results
from REL.entity_disambiguation import EntityDisambiguation
from REL.ner import Cmns, load_flair_ner

from wikimapper_mine import WikiMapper

t0 = time.clock()
t0_wall = time.time()
month = sys.argv[1]
year = sys.argv[2]
split = sys.argv[3]

date=datetime.date(int(year),int(month),1)
date_str = date.strftime('%Y-%m')

#date_str="TEST"
if not os.path.exists(f"selfdata/processed4/{date_str}"):
    os.makedirs(f"selfdata/processed4/{date_str}")
    os.makedirs(f"logs/step4/{date_str}")

logging.basicConfig(filename=f'logs/step4/{date_str}/{split}-GPU.log', level=logging.DEBUG)

base_url = "/home/fvd442/"
wiki_version = "wiki_2019"

with open('politician_data_final.json') as f:
    wikidata = json.load(f)

IDs = []
for politician in wikidata:
    IDs.append(politician['id'])
IDs = set(IDs) # list of politicians covered!

logging.debug("POLITICIANS LOADED")

mention_detection = MentionDetection(base_url, wiki_version)
tagger_ner = load_flair_ner("ner-fast")
#tagger_ngram = Cmns(base_url, wiki_version, n=5)

config = {
    "mode": "eval",
    "model_path": "ed-wiki-2019",
}

model = EntityDisambiguation(base_url, wiki_version, config,reset_embeddings=True)

logging.debug("ENTITY LINKER LOADED")

mapper = WikiMapper("index_enwiki-20190420.db")

logging.debug("MAPPER LOADED")

ner_walltime = 0
ner_processtime = 0
predict_walltime = 0
predict_processtime = 0
process_walltime = 0
process_processtime = 0
map_walltime =0
map_processtime = 0

logging.debug(str(time.clock() - t0))

def NEL(text):
     global ner_walltime
     global ner_processtime
     global predict_walltime
     global predict_processtime
     global process_walltime
     global process_processtime
     global map_walltime
     global map_processtime
     DOC = {"span": [text, []]}
     ckpt_wall = time.time()
     ckpt_process = time.clock()
     mentions_dataset, n_mentions = mention_detection.find_mentions(DOC,tagger_ner)
     ner_walltime += (time.time() - ckpt_wall)
     ner_processtime += (time.clock() - ckpt_process)
     ckpt_wall = time.time()
     ckpt_process = time.clock()
     predictions, timing = model.predict(mentions_dataset)
     predict_walltime += (time.time() - ckpt_wall)
     predict_processtime += (time.clock() - ckpt_process)
     ckpt_wall = time.time()
     ckpt_process = time.clock()
     results = process_results(mentions_dataset,predictions, DOC)
     process_walltime += (time.time() - ckpt_wall)
     process_processtime += (time.clock() - ckpt_process)
     ckpt_wall = time.time()
     ckpt_process = time.clock()     
     NEL_output = []
     name_output = []
     if "span" not in results.keys():
          return [np.nan,np.nan]
     for result in results["span"]:
          if result[4] > 0.3:
               wiki_id = mapper.title_to_id(result[3])
               if wiki_id in IDs:
                    NEL_output.append(wiki_id)
                    name_output.append(result[2])
     map_walltime += (time.time() - ckpt_wall)
     map_processtime += (time.clock() - ckpt_process)
     # ENSURE MAPPING OF INDEX
     inds = []
     seen = set()
     for i, ele in enumerate(name_output):
          if ele not in seen:
               inds.append(i)
          seen.add(ele)
     name_output = list(seen)
     NEL_output = list(np.array(NEL_output)[inds])
     if len(NEL_output) == 0:
          return [np.nan,np.nan]
     else:
          return [NEL_output, name_output]
     #return list(set(NEL_output)), list(set(name_output))

def NEL_series(series):
     DOC = {}
     for i in range(series.shape[0]):
          DOC["span_" + str(i)] = [series.iloc[i], []]
     mentions_dataset, n_mentions = mention_detection.find_mentions(DOC,tagger_ner)
     predictions, timing = model.predict(mentions_dataset)
     results = process_results(mentions_dataset,predictions, DOC)
     output = []
     for doc in results.keys():
          doc_out = []
          for result in results[doc]:
               if result[4] > 0.5:
                    wiki_id = mapper.title_to_id(result[3])
                    if wiki_id in IDs:
                         doc_out.append(wiki_id)
          output.append(list(set(doc_out)))
     #return output

def explode(df, lst_cols, fill_value=''):
    # make sure `lst_cols` is a list
    if lst_cols and not isinstance(lst_cols, list):
        lst_cols = [lst_cols]
    # all columns except `lst_cols`
    idx_cols = df.columns.difference(lst_cols)

    # calculate lengths of lists
    lens = df[lst_cols[0]].str.len()

    if (lens > 0).all():
        # ALL lists in cells aren't empty
        return pd.DataFrame({
            col:np.repeat(df[col].values, df[lst_cols[0]].str.len())
            for col in idx_cols
        }).assign(**{col:np.concatenate(df[col].values) for col in lst_cols}) \
          .loc[:, df.columns]
    else:
        # at least one list in cells is empty
        return pd.DataFrame({
            col:np.repeat(df[col].values, df[lst_cols[0]].str.len())
            for col in idx_cols
        }).assign(**{col:np.concatenate(df[col].values) for col in lst_cols}) \
          .append(df.loc[lens==0, idx_cols]).fillna(fill_value) \
          .loc[:, df.columns]

# I NEED TO MAKE A SAMPLE DATASET HERE
df = pd.read_csv(f"selfdata/processed3/{date_str}/{date_str}_{split}.csv")
logging.debug("SAMPLE TEXT LOADED")
df.dropna(subset=['body'],inplace=True)
logging.debug(df.shape[0])
#df['NEL'] = np.nan
#df['Names'] = np.nan
df['NEL'], df['Names'] = zip(*df.body.apply(lambda x: NEL(x)))
df.dropna(subset=['NEL'],inplace=True)
#df = df[~df.NEL.str.len().eq(0)]
logging.debug("Relevant comments: " + str(df.shape[0]))
df = explode(df, ['NEL','Names'])
logging.debug("Number of entities: " + str(df.shape[0]))

# Final processing of text
df['body'] = df.apply(lambda row: row['body'].replace(row['Names'],"[NAME]"),axis=1)
df['body'] = df['body'].str.replace("^\['", "", regex=True)
df['body'] = df['body'].str.replace("'\]$", "",regex=True)


logging.debug("SAVING")
df.to_csv(f"selfdata/processed4/{date_str}/{split}.csv")
t1 = time.clock()
t1_wall = time.time()
logging.debug("Total Process time: " + str(t1-t0))
logging.debug("Wall time: " + str(t1_wall-t0_wall))
logging.debug("NER Process: " + str(ner_processtime))
logging.debug("NER Wall: " + str(ner_walltime))
logging.debug("Predict Process: " + str(predict_processtime))
logging.debug("Predict Wall: " + str(predict_walltime))
logging.debug("Process Process: " + str(process_processtime))
logging.debug("Process Wall: " + str(process_walltime))
logging.debug("Mapping Process: " + str(map_processtime))
logging.debug("Mapping Wall: " + str(map_walltime))
