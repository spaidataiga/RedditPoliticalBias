import pandas as pd
import datetime

year = 2019
months = [3,4,5,6,7,8,9,10,11,12]

for month in months:
	date=datetime.date(year,month,1)
	date_str = date.strftime('%Y-%m')
	df = pd.read_csv(f"selfdata/comments/{date_str}_data.csv")
	print(df.shape)
	df2 = pd.read_csv(f"selfdata/comments/{date_str}-2.csv")
	df_new = pd.concat([df,df2])
	print(df_new.shape)
	df_new.to_csv(f"selfdata/comments/{date_str}.csv")

	# Now posts

	df = pd.read_csv(f"selfdata/posts/{date_str}_posts.csv")
	print(df.shape)
	df2 = pd.read_csv(f"selfdata/posts/{date_str}_posts2.csv")
	df_new = pd.concat([df,df2])
	print(df_new.shape)
	df_new.to_csv(f"selfdata/posts/{date_str}.csv")

