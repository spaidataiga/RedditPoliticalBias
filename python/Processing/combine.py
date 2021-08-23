import glob
import pandas as pd

dfs = []
for file in glob.glob("selfdata/final/train/201*.csv"):
    dfs.append(pd.read_csv(file).drop(columns=["Unnamed: 0"]))
df = pd.concat(dfs)

df.dropna(subset=['subreddit'], inplace=True)
df = df[~(df.subreddit.str.startswith("Q")) | ~(df.subreddit.str.startswith("["))]

df.to_csv("selfdata/final/train/total.csv", index=False)
