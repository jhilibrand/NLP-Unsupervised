# NLP and Unsupervised Learning Project Proposal 

Question: 
Is there insight to be gained about retail flow from the WallStreetBets subreddit? 

Data Description: 
My primary data set will be top posts from the WallStreetBets subreddit and Yahoo Finance. I aim to pull two primary sets of data: 
1) the top posts *that have tickers in the title* over H1 2021. With this data I aim to overlay the Yfinance library and sentiment analysis to see if there is any predictive modeling that is possible. 
2) the top X posts from H1 2021, without the ticker condition. For this set, I am to do a basic topic modeling to understand what drives teh discssion on this page. 


Tools/MVP: 
I will be scraping Reddit using PushShift and PRAW, and I aim to overlay the Yfinance library as another set of features. I hope to use parts of the NTLK toolkit to run sentiment analysis on Reddit text. An MVP would consist of basic sentiment analysis across a large set of ticker posts. 
