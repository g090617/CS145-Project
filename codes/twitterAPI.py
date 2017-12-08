from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import json
import datetime

consumer_key = "v7PBrhQmnQAnMNkfFYwKRZZ3t"
consumer_secret = "3HO5D8WJqszD6jXvUmLMXQk44zu8vXiQM2eo3QikYcrMC4IkDe"
access_token = "2360536770-fwFoH2gcoxD8aX8gJCdHXHMpfFH1cRUpgqoMZud"
access_token_secret = "c9iT2FJlfe6vv01Tn8xQJQIZkyXEzUfJJaky8J4uPRBjo"


class stdOutListener(StreamListener):

    def on_data(self, data):
        tweet = json.loads(data)

        if 'text' in tweet and 'retweeted' in tweet:
            if not tweet['retweeted'] and \
                            'RT @' not in tweet['text'] and \
                            tweet['lang'] == "en":
                print(tweet['text'], '\n')
                print("=============================")

                fileString = 'data/' + str(datetime.datetime.now().date()) + '-' +\
                             str(datetime.datetime.now().hour) + '.json'
                with open(fileString, 'a') as f:
                    f.write(tweet['text'] + '\n')
                    # f.write(data)
                    return True

    def on_error(self, status):
        print('status', status)

if __name__ == '__main__':

    l = stdOutListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    
    stream = Stream(auth, l)
    stream.filter(track=["microsoft",
                         "Microsoft",
                         "Azure",
                         "XBox",
                         "windows",
                         "Windows",
                         "Bing",
                         "bing",
                         "surface",
                         "Visual Studio",
                         "DirectX",
                         "SQL"], async=True)








