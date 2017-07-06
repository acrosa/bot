#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import ConfigParser
import json
import sys
from expiringdict import ExpiringDict
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
from tweepy import API

import os
import pickle
import tflearn
from tflearn.data_utils import *

# read configuration file (.twitter)
config = ConfigParser.ConfigParser()
config.read('.twitter')
reload(sys)
sys.setdefaultencoding("utf-8")

# store Twitter specific credentials
consumer_key = config.get('apikey', 'key')
consumer_secret = config.get('apikey', 'secret')
access_token = config.get('token', 'token')
access_token_secret = config.get('token', 'secret')
stream_rule = config.get('app', 'rule')
account_screen_name = config.get('app', 'account_screen_name').lower() 
account_user_id = config.get('app', 'account_user_id')

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
twitterApi = API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
path = "./tweets.txt"

class ReplyToTweet(StreamListener):

    def __init__(self):
      self.__max_per_user = 2 # max replies per user within an hour
      self.__rate_limits_per_user = ExpiringDict(max_len=1000, max_age_seconds=(1 * 60))
      self.initialize_model()

    def initialize_model(self):
      char_idx_file = 'char_idx.pickle'
      maxlen = 25

      char_idx = None
      if os.path.isfile(char_idx_file):
        print('Loading previous char_idx')
        char_idx = pickle.load(open(char_idx_file, 'rb'))

      X, Y, char_idx = textfile_to_semi_redundant_sequences(path, seq_maxlen=maxlen, redun_step=3, pre_defined_char_idx=char_idx)

      g = tflearn.input_data([None, maxlen, len(char_idx)])
      g = tflearn.lstm(g, 512, return_seq=True)
      g = tflearn.dropout(g, 0.5)
      g = tflearn.lstm(g, 512, return_seq=True)
      g = tflearn.dropout(g, 0.5)
      g = tflearn.lstm(g, 512)
      g = tflearn.dropout(g, 0.5)
      g = tflearn.fully_connected(g, len(char_idx), activation='softmax')
      g = tflearn.regression(g, optimizer='adam', loss='categorical_crossentropy',
                             learning_rate=0.01)

      m = tflearn.SequenceGenerator(g, dictionary=char_idx,
                              seq_maxlen=maxlen,
                              clip_gradients=5.0,
                              checkpoint_path='model_tweets')
      # Load the model
      m.load("model.tfl")
      self.__text_model = m

    def response(self, text):
      seed = random_sequence_from_textfile(path, maxlen)
      # predict 90 characters (less than 140 chars), change temperature to affect variance on the generation
      text = self.__text_model.generate(90, temperature=1.0, seq_seed=seed)
      return text.lower()

    def on_data(self, data):
      if not data:
        return

      tweet = json.loads(str(data).strip())
      retweeted = tweet.get('retweeted')
      from_self = tweet.get('user', {}).get('id_str','') == account_user_id

      # if retweeted is not None and not retweeted and not from_self:
      if retweeted is not None and not retweeted:
          tweetId = tweet.get('id_str')
          screenName = tweet.get('user', {}).get('screen_name')
          tweetText = tweet.get('text')

          # check if we already replied to this user and when
          # rate limit replies per user
          # update rate for this user
          existing_rate = int(0 if self.__rate_limits_per_user.get(screenName) is None else self.__rate_limits_per_user.get(screenName))
          self.__rate_limits_per_user[screenName] = existing_rate + 1

          # check rate limit, and reply accordingly
          if (existing_rate == self.__max_per_user):
            replyText = '@' + screenName + ' lo siento, pero solo 2 respuestas por hora 🤖👮'
            twitterApi.update_status(status=replyText, in_reply_to_status_id=tweetId)
            print("replied to this user, saying he/she should wait for an hour.")
            return
          elif (existing_rate > self.__max_per_user):
            print("*already* replied to this user, he/she should wait for an hour, ignoring.")
            return

          replyText = '@' + screenName +" "+ self.response("")
          twitterApi.update_status(status=replyText, in_reply_to_status_id=tweetId)
          return

    def on_error(self, status):
      print(status)

if __name__ == '__main__':
    streamListener = ReplyToTweet()
    twitterStream = Stream(auth, streamListener)
    twitterStream.filter(track=['@'+account_screen_name])

