'''
Created on Feb 7, 2017

@author: Emira Ben abdelkrim
'''
from nltk.sentiment.vader import SentimentIntensityAnalyzer
#from gl.contractions import CONTRACTION_MAP
import nltk
from nltk.corpus import sentiwordnet as swn
import random
#from gl.normalization import remove_special_characters, expand_contractions, expand_contractions, correct_text_generic, remove_repeated_characters, tokenize_text, normalize_comment
import pandas as pd
import sys
#from Cython.Compiler.Naming import retval_cname
import pandas as pd
import pandas.io.sql as psql
import numpy as np
from langdetect import detect
#import sqlalchemy
#from sqlalchemy import create_engine
import re


'''
Created on Feb 2, 2017

@author: Emira Ben abdelkrim
'''


#from gl.contractions import CONTRACTION_MAP
import re, collections
import nltk
from nltk.corpus import wordnet
import string
from nltk.stem import WordNetLemmatizer
from html.parser import HTMLParser
import unicodedata
import os


CONTRACTION_MAP = {
"ain't": "is not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how is",
"I'd": "I would",
"I'd've": "I would have",
"I'll": "I will",
"I'll've": "I will have",
"I'm": "I am",
"I've": "I have",
"i'd": "i would",
"i'd've": "i would have",
"i'll": "i will",
"i'll've": "i will have",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she would",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as",
"that'd": "that would",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what'll've": "what will have",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you would",
"you'd've": "you would have",
"you'll": "you will",
"you'll've": "you will have",
"you're": "you are",
"you've": "you have"
}

stopword_list = nltk.corpus.stopwords.words( 'english' )
stopword_list = stopword_list + ['mr', 'mrs', 'come', 'go', 'get',
                                 'tell', 'listen', 'one', 'two', 'three',
                                 'four', 'five', 'six', 'seven', 'eight',
                                 'nine', 'zero', 'join', 'find', 'make',
                                 'say', 'ask', 'tell', 'see', 'try', 'back',
                                 'also']
wnl = WordNetLemmatizer()
html_parser = HTMLParser()


# tokeninzing text
def tokenize_text(text):
    """
    This function takes a text and splits it into tokens (words)
    """
    tokens = nltk.word_tokenize(text)
    tokens = [token.strip() for token in tokens]
    return tokens


# removing special characters
def remove_special_characters(text):
    tokens = tokenize_text(text)
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    filtered_tokens = filter(None, [pattern.sub(' ', token) for token in tokens])
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text
# remove special characters beforer tokenization
def remove_characters_before_tokenization(sentence, keep_apostrophes=False):
    """
    This function removes special characters before tokenization
    """
    sentence = sentence.strip()
    if keep_apostrophes:
        PATTERN = r'[?|$|&|*|%|@|(|)|~]' # add other characters here to remove them
        filtered_sentence = re.sub(PATTERN, r'', sentence)
    else:
        PATTERN = r'[^a-zA-Z0-9 ]' # only extract alpha-numeric characters
        filtered_sentence = re.sub(PATTERN, r'', sentence)
    return filtered_sentence
# expand characters
def expand_contractions( text, contraction_mapping ):
    """
    This function uses the expaned_match function to find each contraction that matches the regrex patten  in CONTRACTION_MAP.
    On any contraction matching , we substiture it with its corresponding expanded version
    """

    contractions_pattern = re.compile( '({})'.format( '|'.join( contraction_mapping.keys() ) ),
                                      flags=re.IGNORECASE | re.DOTALL )
    def expand_match( contraction ):
        match = contraction.group( 0 )
        first_char = match[0]
        expanded_contraction = contraction_mapping.get( match )\
                                if contraction_mapping.get( match )\
                                else contraction_mapping.get( match.lower() )
        expanded_contraction = first_char + expanded_contraction[1:]
        return expanded_contraction

    expanded_text = contractions_pattern.sub( expand_match, text )
    expanded_text = re.sub( "'", "", expanded_text )
    return expanded_text

# removing stopwords (words that have little or no significance)
def remove_stopwords(tokens):
    """
    This function uses English stopwords from nltk and removes all tokens that correponsd to stopwords.
    We are not using this function as we need to do not remove negations words
    """
    stopword_list = nltk.corpus.stopwords.words('english')
    filtered_tokens = [token for token in tokens if token not in stopword_list]
    return filtered_tokens




## correcting words ##
# correcting repeating charcaters
def remove_repeated_characters(text):
    tokens = tokenize_text(text)
    repeat_pattern = re.compile(r'(\w*)(\w)\2(\w*)')
    match_substitution = r'\1\2\3'
    def replace(old_word):
        if wordnet.synsets(old_word):
            return old_word
        new_word = repeat_pattern.sub(match_substitution, old_word)
        return replace(new_word) if new_word != old_word else new_word
    correct_tokens = [replace(word) for word in tokens]
    correct_text = ' '.join(correct_tokens)
    return correct_text
# correcting spellings
def tokens(text):
    """
    Get all words from the corpus
    """
    return re.findall('[a-z]+', text.lower())
WORDS = tokens(open(os.path.dirname(os.path.abspath(__file__)) + '/big.txt').read())
WORD_COUNTS = collections.Counter(WORDS)
def edits0(word):
    """
    Return all strings that are zero edits away
    from the input word (i.e., the word itself).
    """
    return {word}
def edits1(word):
    """
    Return all strings that are one edit away
    from the input word.
    """
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    def splits(word):
        """
        Return a list of all possible (first, rest) pairs
        that the input word is made of.
        """
        return [(word[:i], word[i:]) for i in range(len(word)+1)]
    pairs = splits(word)
    deletes = [a+b[1:] for (a, b) in pairs if b]
    transposes = [a+b[1]+b[0]+b[2:] for (a, b) in pairs if len(b) > 1]
    replaces = [a+c+b[1:] for (a, b) in pairs for c in alphabet if b]
    inserts = [a+c+b for (a,b) in pairs for c in alphabet]
    return set(deletes+transposes+replaces+inserts)
def edits2(word):
    """Return all strings that are two edits away
    from the input word.
    """
    return {e2 for e1 in edits1(word) for e2 in edits1(e1)}

def known(words):
    """
    Return the subset of words that are actually
    in our WORD_COUNTS dictionary.
    """
    return {w for w in words if w in WORD_COUNTS}

def correct(word):
    """
    Get the best correct spelling for the input word
    """
    # Priority is for edit distance 0, then 1, then 2
    # else defaults to the input word itself.
    candidates = (known(edits0(word)) or known(edits1(word)) or known(edits2(word)) or [word])
    return max(candidates, key=WORD_COUNTS.get)

def correct_match(match):
    """
    Spell-correct word in match,
    and preserve proper upper/lower/title case.
    """
    word = match.group()
    def case_of(text):
    #Return the case-function appropriate
    #for text: upper, lower, title, or just str.:
        return (str.upper if text.isupper() else
        str.lower if text.islower() else
        str.title if text.istitle() else
        str)
    return case_of(word)(correct(word.lower()))

def correct_text_generic(text):
    """
    Correct all the words within a text,
    returning the corrected text.
    """
    tokens = tokenize_text(text)
    correct_tokens = [re.sub('[a-zA-Z]+', correct_match, str(word)) for word in tokens]
    correct_text = ' '.join(correct_tokens)
    return correct_text

def keep_text_characters(text):
    filtered_tokens = []
    tokens = tokenize_text(text)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

def normalize_comment(comment,  only_text_chars=False, tokenize=False):

    normalized_comment = []
    for index, text in enumerate(comment):
        text = text.lower()
        text = expand_contractions(text, CONTRACTION_MAP)
        text = remove_stopwords(text)
        text = remove_repeated_characters(text)
        text = correct_text_generic(text)
        if only_text_chars:
            text = keep_text_characters(text)

        if tokenize:
            text = tokenize_text(text)
            normalized_comment.append(text)
        else:
            normalized_comment.append(text)

    return normalized_comment



if __name__ == '__main__':

    corpus = ["AnthonyJose please be fairlllllllly, you automaticly have Cars in any Asphalt game 😂😂😂 and We, normal players can only dreams for this (without the players, who paying for its) 😂😂 BUT, you is great youtuber and I watching all yours videos I don't hate you﻿"]
    normalised_comment = normalize_comment(corpus[0], True, True)
    print(normalised_comment)
    sys.exit(0)

    # clean corpus
    cleaned_corpus = [remove_characters_before_tokenization(sentence.lower(), keep_apostrophes=True) for sentence in corpus]
    print(cleaned_corpus)
    # expand contractions
    expanded_corpus = [expand_contractions(sentence, CONTRACTION_MAP) for sentence in cleaned_corpus]
    print(expanded_corpus)
    # remove repeated characters
    expanded_corpus_tokens = [tokenize_text(text) for text in expanded_corpus]
    print(expanded_corpus_tokens)
    filtered_list1= [[remove_repeated_characters(tokens) for tokens in sentence_tokens] for sentence_tokens in expanded_corpus_tokens]
    print(filtered_list1)

    print(correct_text_generic('automaticly'))
    filtered_list2 = [[correct_text_generic(str(tokens)) for tokens in sentence_tokens] for sentence_tokens in expanded_corpus_tokens]
    print(filtered_list2)






def convert_emojis_to_text(message):
    new_message = ''
    emoji_list = ['😆', '😀', '😗', '😍', '👺', '👻', '💀', '👽', '😅', '😋', '😎', '😵', '😇', '😊', '💓', '😤', '😬', '😡', '🎮', '👌', '🙌', '👍', '☺', '🍆', '💣', '🎳', '🎱', '⚽', '❤', '😈', '💖', '💕', '👙', '🚘', '🚗', '🚓', '🏁', '😢', '🚮', '👎', '🔚', '🔥', '😭', '🍎', '👫', '💜', '🍌', '🙏', '🏆', '💟', '💝', '👗', '👠', '👑', '💵', '😻', '🐆', '⭐', '👦', '💂', '🙇', '🌈', '👹', '😈', '👏', '😶', '😐', '😉', '😸', '😺', '😣', '💯', '😱', '💌', '🔫', '🔪', '🚲', '😠', '💘', '💗', '😚', '😙', '😘', '👿', '😋', '🚔', '🚓', '🙀', '😜', '😨', '😥', '😒', '😭', '😕', '😴', '⚠', '🙅', '🏍', '🏎', '✌', '💑', '🖒', '💄', '👆', '😛', '🚫', '✨', '💎', '👇', '💋', '💞', '👄', '🙍', '🎉', '💏', '👧', '👩', '💔', '💙', '🏈', '🎆', '🎇', '✔', '💃', '👉', '►', '【', '】', '🔴', '👀', '📲', '💰', '☕', '🐸', '🗣', '💚', '💦', '💥', '💩', '🐮', '⏱', '♀', '🚀', '🎈', '💪', '👊', '🎵', '🎶', '🔊', '🎡', '🎢', '🏰', '⚡', '🍀']
    emoji_dict = {'😺': 'love', '👫': 'love', '🍎': 'cow', '😎': 'love', '🙇': 'cow', '💑': 'love', '😇': 'love', '😵': 'love', '😴': 'hate', '👌': 'love', '】': 'cow', '⚠': 'hate', '😆': 'love', '🌈': 'love', '😐': 'cow', '👄': 'love', '✨': 'cow', '😜': 'love', '🙀': 'hate', '💜': 'love', '💞': 'love', '🎡': 'love', '😒': 'hate', '🔫': 'cow', '😤': 'hate', '😸': 'love', '👺': 'hate', '😢': 'hate', '🍌': 'cow', '😣': 'hate', '🔚': 'hate', '💌': 'love', '🏆': 'love', '✔': 'love', '😉': 'love', '😭': 'hate', '👙': 'love', '🏍': 'cow', '💓': 'love', '💕': 'love', '😋': 'love', '💋': 'love', '👻': 'hate', '😥': 'hate', '💪': 'cow', '🚓': 'cow', '💵': 'cow', '👉': 'cow', '😚': 'love', '🚘': 'cow', '💘': 'love', '🎮': 'cow', '🏰': 'cow', '😬': 'hate', '🎆': 'love', '💎': 'love', '😀': 'love', '🐸': 'cow', '🏎': 'cow', '🎶': 'cow', '👩': 'cow', '⚽': 'cow', '【': 'cow', '👏': 'love', '💀': 'hate', '👦': 'cow', '⏱': 'cow', '👑': 'love', '😈': 'hate', '😕': 'hate', '📲': 'cow', '💚': 'cow', '🔴': 'cow', '💃': 'love', '🚮': 'hate', '💗': 'love', '😱': 'love', '💏': 'love', '😘': 'love', '💔': 'hate', '😛': 'love', '💂': 'cow', '👽': 'cow', '👍': 'love', '👹': 'hate', '👎': 'hate', '😙': 'love', '💄': 'love', '🔊': 'cow', '🎈': 'love', '😻': 'love', '😅': 'love', '💩': 'hate', '🎉': 'love', '🎢': 'love', '🐮': 'cow', '☺': 'love', '🍆': 'love', '🐆': 'cow', '💝': 'love', '🚲': 'cow', '►': 'cow', '☕': 'cow', '👀': 'cow', '🎳': 'cow', '🗣': 'cow', '✌': 'love', '👿': 'hate', '💣': 'love', '♀': 'cow', '🍀': 'love', '⚡': 'love', '🙏': 'cow', '🚔': 'cow', '👊': 'love', '😨': 'hate', '🙌': 'love', '😠': 'hate', '👆': 'love', '😍': 'love', '🙍': 'hate', '🎵': 'cow', '💥': 'love', '🔥': 'love', '😗': 'love', '🏁': 'love', '😶': 'cow', '⭐': 'love', '❤': 'love', '👠': 'cow', '💦': 'cow', '👇': 'hate', '🚀': 'love', '🏈': 'cow', '👗': 'cow', '💯': 'love', '🙅': 'hate', '🎇': 'love', '😊': 'love', '🚗': 'cow', '🖒': 'love', '😡': 'hate', '🔪': 'cow', '🎱': 'cow', '💙': 'love', '💟': 'love', '💰': 'cow', '🚫': 'hate', '💖': 'love', '👧': 'cow'}

    for letter in message:
        if letter in emoji_list:
            new_message += ' ' + emoji_dict[letter] + ' '
        else:
            new_message += letter

    return new_message

def analyze_sentiment_sentiwordnet_lexicon( comment, verbose=False ):
    """
    This function uses the sentiWordNet lexicon for sentiment analysis.
    The sentiWordNet lexicon assigns three sentiment scores to it, including a postive polarity,
    a negative polarity score and an objectivity score
    """

    #does not process comments containing words bigger than 50 characters or else the script crashes and no words are bigger than 50 characters
    if re.search("[^\s]{50,}", comment, flags=re.IGNORECASE):
        return 0

    # # pre-process comment
    comment = normalize_comment( comment )
    # tokenize and POS tag text tokens
    comment_tokens = nltk.word_tokenize( comment )
    tagged_comment = nltk.pos_tag( comment_tokens )

    pos_score = neg_score = token_count = obj_score = 0

    # get wordnet synsets based on POS tags
    # get sentiment scores if synsets are found
    for word, tag in tagged_comment:

        word = word.lower()
        # print( word )
        ss_set = None
        if 'NN' in tag and list( swn.senti_synsets( word, 'n' ) ):

            ss_set = list( swn.senti_synsets( word, 'n' ) )[0]
        elif 'VB' in tag and list( swn.senti_synsets( word, 'v' ) ):

            ss_set = list( swn.senti_synsets( word, 'v' ) )[0]
        elif 'JJ' in tag and list( swn.senti_synsets( word, 'a' ) ):

            ss_set = list( swn.senti_synsets( word, 'a' ) )[0]
        elif 'RB' in tag and list( swn.senti_synsets( word, 'r' ) ):

            ss_set = list( swn.senti_synsets( word, 'r' ) )[0]

        # if senti-synset is found
        if ss_set:
            # add scores for all found synsets
            pos_score += ss_set.pos_score()
            neg_score += ss_set.neg_score()
            obj_score += ss_set.obj_score()
            token_count += 1

    # aggregate final scores
    if obj_score > 0 and pos_score == 0 and neg_score == 0 :
        final_sentiment = 'neutral'
        final_score = 1
        retval = 0
    else :

        final_score = pos_score
        final_sentiment = 'positive'
        retval = 1
        if neg_score > final_score:
            final_score = neg_score
            final_sentiment = "negative"
            retval = -1

    # norm_final_score = round( float( final_score ) / token_count, 2 )


    if verbose:
        norm_obj_score = round( float( obj_score ) / token_count, 2 )
        norm_pos_score = round( float( pos_score ) / token_count, 2 )
        norm_neg_score = round( float( neg_score ) / token_count, 2 )
        norm_final_score = norm_pos_score - norm_neg_score
        # to display results in a nice table
        sentiment_frame = pd.DataFrame( [[final_sentiment, norm_obj_score, norm_pos_score, norm_neg_score, norm_final_score]], columns=pd.MultiIndex( levels=[['SENTIMENT STATS:'], ['Predicted Sentiment', 'Objectivity', 'Positive', 'Negative', 'Overall']], labels=[[0, 0, 0, 0, 0], [0, 1, 2, 3, 4]] ) )
        print ( sentiment_frame )
    return retval

def analyze_sentiment_vader_lexicon( comment, threshold=0.1, verbose=False ):
    """
    This function uses the VADER lexicon for sentiment analysis,
    specially built for analyzing sentiment from social media resources.
    7500 lexical features with various terms, including words, emoticons, and even slang language-
    based tokens (like lol , wtf , nah , and so on)
    Each feature was rated on a scale from "[-4] Extremely Negative" to "[4] Extremely Positive" , with allowance for "[0] Neutral (or Neither, N/A)" .
    """

    if re.search("[^\s]{50,}", comment, flags=re.IGNORECASE):
        return 0

    # pre-process comment
    comment = normalize_comment( comment )
    # analyze the sentiment for comment
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores( comment )
    agg_score = scores['compound']
    pos_score = scores['pos']
    neg_score = scores['neg']
    neu_score = scores['neu']

    # get aggregate scores and final sentiment
    if neu_score > 0 and pos_score == 0 and neg_score == 0 :
        final_sentiment = 'neutral'
        final_score = 1
        retval = 0
    else :
        if agg_score >= threshold:
            final_sentiment = 'positive'
            retval = 1
        else :
            final_sentiment = 'negative'
            retval = -1


    if verbose:
        # display detailed sentiment statistics
        positive = str( round( scores['pos'], 2 ) * 100 ) + '%'
        final = round( agg_score, 2 )
        negative = str( round( scores['neg'], 2 ) * 100 ) + '%'
        neutral = str( round( scores['neu'], 2 ) * 100 ) + '%'
        sentiment_frame = pd.DataFrame( [[final_sentiment, final, positive,
                                        negative, neutral]],
                                        columns=pd.MultiIndex( levels=[['SENTIMENT STATS:'],
                                                                      ['Predicted Sentiment', 'Polarity Score',
                                                                       'Positive', 'Negative',
                                                                       'Neutral']],
                                                              labels=[[0, 0, 0, 0, 0], [0, 1, 2, 3, 4]] ) )
        print( sentiment_frame )

    return  retval


def normalize_comment( comment, stop_words = False):
    #convert emojis to text
    comment = convert_emojis_to_text(comment)

    comment = comment.lower()
    # expand contractions
    comment = expand_contractions( comment, CONTRACTION_MAP )
    # remove repeated characters
    comment = remove_repeated_characters( comment )
    # correct tokens
    comment = correct_text_generic( comment )
    if stop_words :
        comment = remove_special_characters(comment)

    return comment

def getLang( comment ):
    lang = 'NAN'
    try :
        lang = detect( comment )
        return lang
    except :
        return lang
