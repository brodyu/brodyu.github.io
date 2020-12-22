---
layout: post
title:  "Twitter Sentiment Analysis with Natural Language Processing"
date:   2020-03-11 18:05:55 +0300
image:  '/assets/img/Twitter.jpg'
tags:   NLP
---
# Introduction
This project is a quick way to aggregate tweets and perform sentiment analysis. The goal of this project was to learn the overall sentiment (positive, negative, or neutral) of a particular topic on Twitter. Utilizing the official Twitter API and several natural language processing libraries such as TextBlob, we are able to gain insight into general public sentiment quickly and efficiently. Senitment Analysis is a huge field of natural language processing research. The advances in the field are utilized in a wide variety of industries from marketing to finance. 

# Requirements
This library utilizes a number of libraries to automate data capture and analysis. These specific libraries and their function is listed below:
* Twitter API: In order to use Twitter's API, you need to sign up for a developer account and submit a project proposal for the API keys. Click on the link [here](https://developer.twitter.com/en) to sign up for a developer account.
* Tweepy: The Tweepy library is used to authenticate the Twitter API
* TextBlob: is a pretrained NLP model that we will run across our tweets. 

# Implementation
Once you have made a developer account with Twitter, you gain access to pairs of consumer keys and access tokens. Copy and paste your codes as such:

```python
# Consumer API keys:
consumer_key = "___"
consumer_key_secret = "___"
# Access token & access token secret
access_token = "___"
access_token_secret = "___"
```

Once your keys are set, we can autenticate our requests with Tweepy's OAuthHandler method.
```python
auth = tweepy.OAuthHandler(consumer_key, consumer_key_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)
```

Now that our authentication is complete, all we need to do is run Twitter's search method. The line of code below requests a keyword to search for and returns a list of tweets that represents the results:
```python
public_tweets = api.search(q = self.search, count = self.count, result_type = self.result_type, until = self.until, lang = self.lang)
```

In order to gain sentiment on each tweet, we want to run the TextBlob library on each tweet to gain its sentiment score with the sentiment method. TextBlob's sentiment polarity score ranges from -1.0 (negative sentiment) to 1.0 (positive sentiment).
```python
for tweet in public_tweets:
  print(tweet.text)
  analysis = TextBlob(tweet.text)
  print(analysis.sentiment)
  if analysis.sentiment[0]>0:
    print('Positive')
    positive_count += 1
  else:
    print('Negative')
    negative_count += 1
  print("")
```



# Where to find
The link to the GitHub repository that stores our Python implementation can be found [here](https://github.com/brodyu/twitter_sentiment_analysis)


