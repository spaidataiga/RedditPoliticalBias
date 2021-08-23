import datetime as dt
import calendar
import pandas as pd
import os
from dateutil.relativedelta import relativedelta
from pmaw import PushshiftAPI

api = PushshiftAPI()
SOI = ['democrats','europe','ireland']
times = [(2019,2),(2019,3),(2019,4),(2019,5),(2019,6),(2019,7),(2019,8),(2019,9),(2019,10),(2019,11),(2019,12)]
#times = [(2018,7),(2018,8),(2018,9),(2018,10),(2018,11),(2018,12),(2019,1)]

# # To scrape posts. ALREADY DONE
# for month in times:
#     day = dt.datetime(month[0],month[1],1)
#     gen_p = api.search_submissions(after=int(day.timestamp()),
#                             before=int((day+dt.timedelta(calendar.monthrange(day.year, day.month)[1])).timestamp()),
#                             subreddit=SOI,
#                             filter=['id','title', 'url', 'selftext', 'subreddit'])
#     df = pd.DataFrame([submission for submission in gen_p])
#     df.to_csv("selfdata/NEW/posts/" + day.strftime("%Y-%m") + "_posts.csv")
#     print(day)

for month in times:
    day = dt.datetime(month[0],month[1],1)
    gen_c = api.search_comments(after=int(day.timestamp()),
                                before=int((day+dt.timedelta(calendar.monthrange(day.year, day.month)[1])).timestamp()),
                                subreddit=SOI,
                                filter=['id', 'body', 'link_id', 'parent_id','subreddit'])
    df = pd.DataFrame([submission for submission in gen_c])
    df.to_csv("selfdata/comments/" + day.strftime("%Y-%m") + "-2.csv")
    print(day)
