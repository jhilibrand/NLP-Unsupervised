import pandas as pd 
import numpy as np 
import requests 
import praw 
from collections import Counter 
#nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import regex as re
from  psaw import PushshiftAPI
import pickle
from datetime import datetime
import string
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import NMF, TruncatedSVD, LatentDirichletAllocation
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
import streamlit as st 

word = st.text_input("Input a phrase:", value = 'citadel')

with open('/Users/jenniferhilibrand/Metis/NLP/reddit_topic_matrix.pickle', 'rb') as to_read: 
    reddit_topic_matrix_df = pickle.load(to_read)



sia = SentimentIntensityAnalyzer()
def vader_sentiment(review):
    vs = sia.polarity_scores(review)
    for senti,score in vs.items(): 
        if score > 0:
            return(senti)



my_stop_words = ['www', 'reddit', 'com', 'https', '']
stop_words = ENGLISH_STOP_WORDS.union(my_stop_words)


reddit_topic_matrix_df2 = reddit_topic_matrix_df
clean_comments = list(reddit_topic_matrix_df2['clean_comments'])
clean_comments.append(word)
corpus_new=clean_comments

vectorizer = CountVectorizer(stop_words = stop_words)
doc_word = vectorizer.fit_transform(corpus_new)
tfidf = TfidfVectorizer(stop_words=stop_words)
reddit_word_matrix2 = tfidf.fit_transform(corpus_new)
nmf_model = NMF(5)
doc_topic = nmf_model.fit_transform(doc_word)
topic_word = pd.DataFrame(nmf_model.components_.round(5),
             index = ["component_1","component_2", "component_3","component_4","component_5"],
             columns = tfidf.get_feature_names())
nmf_model.fit(reddit_word_matrix2)
reddit_topic_matrix =  nmf_model.transform(reddit_word_matrix2)
reddit_topic_matrix_df2 = pd.DataFrame(reddit_topic_matrix).add_prefix('topic_')



reddit_topic_matrix_df2['comments']=clean_comments
distance_matrix=reddit_topic_matrix_df2 .set_index('comments')

input_row = distance_matrix.tail(1)

dist_list=[]
for index, row in distance_matrix.iterrows():
    dist_list.append(np.linalg.norm(row - input_row))
    
recommender_df = pd.DataFrame(dist_list)
recommender_df['comments']=clean_comments
top = recommender_df.sort_values(by=[0])

similar_comments = top['comments'][1:101]
sentiment_similar=[]
for i in similar_comments:
    sentiment_similar.append(vader_sentiment(i))
senti_count=Counter(sentiment_similar)
st.write(senti_count['neg'], '% negative')
st.write(senti_count['pos'], '% positive')
st.write('10 Most Similar Comments')
for i in similar_comments[0:10]:
    st.write(i)