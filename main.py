import json
import streamlit as st
import requests
import nltk
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
import main_functions
from pprint import pprint
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from bokeh.plotting import figure
from PIL import Image
import time
import numpy as np
import pandas as pd
import plotly.express as px

api_key_dict = main_functions.read_from_file("JSON_Files/api_key.json")
api_key = api_key_dict["my_key"]

def api_call(api_type, type, time_period):
    if (api_type == 'stories'):
        url = 'https://api.nytimes.com/svc/topstories/v2/'+ type + '.json?api-key=' + api_key
        response = requests.get(url).json()
        main_functions.save_to_file(response, "JSON_Files/response.json")
    elif (api_type == 'articles'):
        url = 'https://api.nytimes.com/svc/mostpopular/v2/' + type + '/' + time_period + '.json?api-key=' + api_key
        response = requests.get(url).json()
        main_functions.save_to_file(response, "JSON_Files/article_response.json")


def generate_data(type, response_type):
    if (response_type == 'stories'):
        response = main_functions.read_from_file("JSON_Files/response.json")
    elif (response_type == 'articles'):
        response = main_functions.read_from_file("JSON_Files/article_response.json")
    str1 = ""
    for i in response["results"]:
        str1 = str1 + i["abstract"]
    sentences = sent_tokenize(str1)
    words = word_tokenize(str1)
    fdist = FreqDist(words)
    words_no_punc = []
    for w in words:
        if w.isalpha():
            words_no_punc.append(w.lower())
    fdist2 = FreqDist(words_no_punc)
    stopwording = stopwords.words("english")
    cleanwords = []
    for w in words_no_punc:
        if w not in stopwording:
            cleanwords.append(w)
    fdist3 = FreqDist(cleanwords)
    fdist3.most_common(10)
    wordcloud = WordCloud().generate(str1)
    save_image = wordcloud.to_file('IMAGE_Files/cloud.png')
    save_image
    data = fdist3.most_common(10)
    ind = np.arange(len(data))
    names, values = zip(*data)
    if (type == 'graph'):
        dframe = pd.DataFrame({
            'words': names,
            'count': values
        })
        fig = px.line(dframe, x='words', y='count', title='Top 10 Most Common Words')
        st.plotly_chart(fig)
    if (type == 'wordcloud'):
        st.image(Image.open('IMAGE_Files/cloud.png'), caption='Wordcloud', use_column_width = True)

st.title('Project 1')
st.header('Part A - The Stories API')
st.empty()
st.markdown('**1 - Topic Selection**')
name = st.text_input('Please enter your name')
option = st.selectbox('Select a Topic of your intrest', options=('', 'arts', 'automobiles', 'books', 'business',
                                                               'fashion', 'food', 'health', 'home', 'insider',
                                                               'magazine', 'movies', 'nyregion', 'obituaries',
                                                               'opinion', 'politics', 'realestate', 'science', 'sports', 'sundayreview',
                                                               'technology', 'theater', 't-magazine', 'travel', 'upshot', 'us', 'world'), index=0)


if name and option:
    st.write('Hey', name, 'You selected:', option, 'Topic')
    api_call('stories', option, None)
    st.markdown('**2 - Frequency Distribution**')
    frequency_checkbox = st.checkbox('Click here to generate frequency distribution')
    if frequency_checkbox:
        generate_data('graph', 'stories')
    wordcloud_checkbox = st.checkbox('Click here to generate worldcloud')
    if wordcloud_checkbox:
        generate_data('wordcloud', 'stories')
else:
    st.empty()

st.header('Part B - Most Popular Articles')
set_of_articles = st.selectbox('Select your preffered set of articles', options=('', 'shared', 'emailed', 'viewed'), index=0)
set_period = st.selectbox('Select the period of time', options=('', '1', '7', '30'), index=0)
if set_of_articles and set_period:
    api_call('articles', set_of_articles, set_period)
    time.sleep(1)
    generate_data('wordcloud', 'articles')


