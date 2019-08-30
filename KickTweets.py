import tweepy
import pandas as pd     # To handle data
import numpy as np      # For number computing

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

tweets = extractor.search(q="The Faceless+kickstarter", lang="en", count=100, tweet_mode="extended", show_user='true')
twitter_text = []
twitter_fav_counts = []
twitter_retweet_counts = []
print("Number of tweets extracted: {}.\n".format(len(tweets)))
    
for tweet in tweets:
    if 'retweeted_status' in tweet._json:
        twitter_text.append(tweet._json['retweeted_status']['full_text'])
        twitter_fav_counts.append( tweet._json['favorite_count'])
        twitter_retweet_counts.append( tweet._json['retweet_count'])
        #print(tweet_dataframe)
    else:
        twitter_text.append( tweet._json['full_text'])
        twitter_fav_counts.append(tweet._json['favorite_count'])
        twitter_retweet_counts.append( tweet._json['retweet_count'])
   
d = {'Tweets': twitter_text,
     'Like_count': twitter_fav_counts,
     'Retweet_Counts' : twitter_retweet_counts
     }

tweet_dataframe = pd.DataFrame(d)
print(tweet_dataframe)
tweet_dataframe.to_csv(r'E:\Masters\Project\Kickstarter\Tweets_198X.csv', index=None, sep=',',encoding='utf-8', mode='a')
    #analysis = TextBlob(tweet.full_text)
    #print(analysis.sentiment)
    '''
