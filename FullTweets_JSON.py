import tweepy
import pandas as pd     # To handle data
import numpy as np      # For number computing
import re

#Consumer key from twitter api
CONSUMER_KEY    = 'CONSUMER_KEY'
#Consumer Secret from twitter api
CONSUMER_SECRET = 'CONSUMER_SECRET'
# Access Token from Twitter API:
ACCESS_TOKEN  = 'ACCESS_TOKEN'
ACCESS_SECRET = 'ACCESS_SECRET'

from credentials import *  # This will allow us to use the keys as variables
from textblob import *

# API's setup:
def twitter_setup():
    """
    Utility function to setup the Twitter's API
    with our access keys provided.
    """
    # Authentication and access using keys:
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)

    # Return API with authentication:
    api = tweepy.API(auth)
    return api

extractor = twitter_setup()

tweets = extractor.search(q="kickstarter", lang="en", count=100, maxResults=500,tweet_mode="extended", show_user='true')

favorite_count  =[]
retweet_count =[]
source  =[]
text =[]
user_name  =[]
hashtag = []
polarity = []
subjectivity = []
#result_type=[]

print("Number of tweets extracted: {}.\n".format(len(tweets)))
    
for tweet in tweets:
    favorite_count.append(tweet._json['favorite_count'])
    retweet_count.append(tweet._json['retweet_count'])
    source.append(tweet._json['source'])
    text.append(tweet._json['full_text'])
    #result_type.append(tweet._json['metadata']['result_type'])
        
    analysis = TextBlob(tweet._json['full_text'])
    polarity.append(analysis.polarity)
    subjectivity.append(analysis.subjectivity)
    
d = {'FavoriteCount':favorite_count,
     'RetweetCount':retweet_count,
     'Source':source,
     'Text':text,
     'Polarity' : polarity,
     'Subjectivity' : subjectivity
     #'result_type': result_type
     }

tweet_dataframe = pd.DataFrame(d)
#print(tweet_dataframe)
tweet_dataframe.to_csv(r'E:\Masters\Project\Kickstarter\Data\DesignTech.csv', index=None, sep=',',encoding='utf-8', mode='a')
