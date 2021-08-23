import pandas as pd
import sys
import datetime

month = sys.argv[1]
year = sys.argv[2]

date=datetime.date(int(year),int(month),1)
date_str = date.strftime('%Y-%m')

df = pd.read_csv(f'selfdata/final/{date_str}.csv')
df.dropna(how='all', inplace=True)
df.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'], inplace=True)
print("SIZE", df.shape[0])

pst_1 = pd.read_csv(f"selfdata/NEW/posts/{date_str}_posts.csv")
pst_2 = pd.read_csv(f"selfdata/NEW/posts/{date_str}_posts2.csv")
psts = pd.concat([pst_1,pst_2])

cmt_1 = pd.read_csv(f"selfdata/NEW/comments/{date_str}.csv")
cmt_2 = pd.read_csv(f"selfdata/NEW/comments/{date_str}22.csv")
cmts = pd.concat([cmt_1,cmt_2])

print(df[df['subreddit'].isna()].shape) # how many are missing a subreddit?

interim = df.merge(cmts[['id','link_id']], on='id')
interim.drop(columns=['subreddit'], inplace=True)
interim.dropna(how='all', inplace=True)

interim['link_id'] = interim['link_id'].map(lambda x: x.split('_')[1])

psts = psts[['id','subreddit']]
psts.columns = ['link_id', 'subreddit']

interim = interim.merge(psts, on='link_id')

interim.drop(columns=['link_id'], inplace=True)
df_half = df.dropna(subset=['subreddit'])
df = pd.concat([df_half, interim])
print("no subreddit", df[df['subreddit'].isna()].shape[0])
print("TOTAL", df.shape[0])
df.to_csv(f"selfdata/final/fixed/{date_str}.csv")
