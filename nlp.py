#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 17:21:14 2020

@author: quantiphi
"""

import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
nlp = spacy.load("en_core_web_sm")
from bs4 import BeautifulSoup
analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    sent_score = []
    main_sentiment = analyzer.polarity_scores(text)
    sentences = text.split('.')
    for s in sentences:
        score = analyzer.polarity_scores(s)
        sent_score.append([s, 'score :' + str(score)])
    return main_sentiment, sent_score

def url_to_string(url):
    res = requests.get(url)
    html = res.text
    soup = BeautifulSoup(html, 'html5lib')
    for script in soup(["script", "style"]):
        script.extract()
    return soup.get_text()

def spacy(text, nlp):
    doc = nlp(text)
    
    tokens = [token.text for token in doc]
    all_ents = []
    for tk in tokens:
        tk_doc = nlp(tk)
        tk_ent = [[ent.text, ent.label_] for ent in tk_doc.ents]
        if tk_ent != []:
            all_ents.append(tk_ent)
    return all_ents, tokens