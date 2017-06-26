# Preprocessing of tweets

Tweets from the twitter sentiment dataset are encoded in plain ascii, and are all in English. This is not enough for our purposes, for for now it will have to do. The training data for twitter is contained in the table 'twitter_sentiment_dataset', which has 4 columns:

 - `id` is a unique primary key
 - `sentiment` is a $\{0,1\}$-valued column, indicating whether the tweet is positive ($1$) or negative ($1$). The sentiment classification we will use will have $4$ possible values: positive, negative, neutral or irrelevants
 - `text` contains the raw text of the tweet, encoded in utf-8
 - `sanitized_text` contains the texrt of the tweet, with all usernames @username are replaces by the token "@\<!USER!\>", and all urls are trplaced bu the token "\<!URL!\>"

`sanitized_text` is the column we will use for training.
