import sys
import logging
import spacy
import pandas as pd
import time
import numpy as np
import re
import datetime
from dateutil.relativedelta import relativedelta

def NE_present(comment):
    text = nlp(comment)
    NEs = any([e.label_ == 'PERSON' for e in text.ents])
    if NEs:
        return True
    else:
        return False

month = sys.argv[1]
year = sys.argv[2]

date=datetime.date(int(year),int(month),1)
date_str = date.strftime('%Y-%m')
logging.basicConfig(filename=f'logs/{date_str}.log', level=logging.DEBUG)
logging.debug(spacy.__version__)
logging.debug(date_str)
t00 = time.clock()
nlp = spacy.load('en_core_web_md')
logging.debug("Spacy and Neural Coref loaded.")

# Load data
# Load data
cmts = pd.read_csv(f'selfdata/comments/{date_str}.csv')
print("Comments loaded.")
cmts.drop_duplicates('id', inplace=True)
cmts.set_index('id',inplace=True)
posts = pd.read_csv(f"selfdata/posts/{date_str}_posts.csv",index_col='id')

#Now add in non-archived posts (aka posts from the last 6 months)
for i in range(1,7):
        new_day = date + relativedelta(months=-i)
        posts = pd.concat((posts,pd.read_csv(f"selfdata/posts/{new_day.strftime('%Y-%m')}_posts.csv",index_col='id')))

logging.debug("Posts loaded.")

SOI = ['politics', 'uspolitics', 'canada', 'The_Donald', 'feminisms', 'SocialDemocracy', 'Liberal', 'news',
       'TwoXChromosomes', 'teenagers', 'socialism', 'Libertarian', 'Conservative', 'alltheleft', 'neoliberal', 'Republican', 'MensRights',
      'unitedkingdom', 'australia', 'newzealand', 'india','democrats','europe',
	'ireland']

#isolate to English-speaking subreddits
#cmts = cmts[cmts['subreddit'].isin(SOI)]
#posts = posts[posts['subreddit'].isin(SOI)]

# only keep comments on collected posts
links = posts.index.unique().tolist()
cmts["link_id"] = cmts["link_id"].str.split("_", n = 1, expand = True)[1] 
df = cmts.query('link_id in @links')

# only keep unique comments
df.drop_duplicates('body', inplace=True)

del cmts

# Seperate the parent_id comment to determine what the comment is a response to and the id
df["to_type"] = df["parent_id"].str.split("_", n = 1, expand = True)[0] 
df["to_id"] = df["parent_id"].str.split("_", n = 1, expand = True)[1] 
df.drop(columns="parent_id", inplace=True)


logging.debug("Processing text")
# Process the text -- refer to https://www.reddit.com/r/raerth/comments/cw70q/reddit_comment_formatting/

df.dropna(subset=['body'], inplace=True) # Drop all NaN comments (aka deleted)
# df = df[~df.body == 'removed']
# df = df[~df.body == 'deleted']

# Remove URLs and formatting
df['body'] = df['body'].str.replace(r'\(https?:\/\/.*\)', "URL").str.replace(r'[\[\]]', "")

# Remove all posts from bots
df = df[~df['body'].str.contains(r"I'?( a)?m a bot")] #Generally, bots clearly state they are a bot

# Remove all text formatting (italics / bold)
df['body'] = df['body'].str.replace(r"[_|\*|~|\^|\|]","")
df['body'] = df['body'].str.replace("&lt;","") # drop lists
df['body'] = df['body'].str.replace("&gt;","") # drop quotes?

# Convert all remaining URLs to "URL"
df['body'] = df['body'].str.replace(r'https?:\/\/.*', "URL")

# Add context (from link)
df.dropna(subset=['body'], inplace=True) #remove if no comment
posts.dropna(subset=['title'], inplace=True) #remove if no info

post_reference = posts['title'].to_dict()
post_named = posts['title'].map(NE_present).to_dict()
df['post_named'] = df['link_id'].map(post_named)
df['context'] = df['link_id'].map(post_reference)
del post_reference
del posts

# Keep only comments with named entities
logging.debug("Searching for named entities!")
df['cmt_named'] = df['body'].map(NE_present)
df['named'] = df.loc[:,['cmt_named','post_named']].any(axis=1)

logging.debug("Saving")
# Save the data for coreference resolution with context
df.to_csv(f"selfdata/processed/{date_str}.csv")
logging.debug("Done.")
logging.debug("Total time elapsed: " + str(time.clock() - t00))
