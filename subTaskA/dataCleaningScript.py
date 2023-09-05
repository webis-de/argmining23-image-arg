#Importing required libraries pandas for data frames and regular expression library re
import re 
import pandas as pd
import neattext as nt 
from random import randint
import os
import numpy as np
import random
import re
import wordninja

def replace_and_split_hashtags(text):
    def modify_hashtag(match):
        hashtag = match.group(1)
        modified_hashtag = re.sub(r'(?<=[a-z])(?=[A-Z])|_', ' ', hashtag)
        return f'{" ".join(wordninja.split(modified_hashtag))}'

    modified_text = re.sub(r'#(\w+)', modify_hashtag, text)
    return modified_text

def appriviation_converter(appriviated_text: str)-> str:
    '''This function takes text check if there are apprivations like i'm  or he's and convert it to its regular form'''
    appriviated_text = re.sub(r"i'm", "i am", appriviated_text)
    appriviated_text = re.sub(r"he's", "he is", appriviated_text)
    appriviated_text = re.sub(r"she's", "she is", appriviated_text)
    appriviated_text = re.sub(r"that's", "that is", appriviated_text)        
    appriviated_text = re.sub(r"what's", "what is", appriviated_text)
    appriviated_text = re.sub(r"where's", "where is", appriviated_text) 
    appriviated_text = re.sub(r"\'ll", " will", appriviated_text)  
    appriviated_text = re.sub(r"\'ve", " have", appriviated_text)  
    appriviated_text = re.sub(r"\'re", " are", appriviated_text)
    appriviated_text = re.sub(r"\'d", " would", appriviated_text)
    appriviated_text = re.sub(r"\'ve", " have", appriviated_text)
    appriviated_text = re.sub(r"won't", "will not", appriviated_text)
    appriviated_text = re.sub(r"don't", "do not", appriviated_text)
    appriviated_text = re.sub(r"did't", "did not", appriviated_text)
    appriviated_text = re.sub(r"can't", "can not", appriviated_text)
    appriviated_text = re.sub(r"it's", "it is", appriviated_text)
    appriviated_text = re.sub(r"couldn't", "could not", appriviated_text)
    appriviated_text = re.sub(r"have't", "have not", appriviated_text)

    unapriviated_text = appriviated_text
    return unapriviated_text

def clean_text(uncleaned_text : str ) -> str :
    uncleaned_text = replace_and_split_hashtags(uncleaned_text)
    uncleaned_text = appriviation_converter(uncleaned_text)
   
    '''This function takes uncleaned_text and clean the text using the neatttext library and returns the cleaned text'''
    text_frame = nt.TextFrame(uncleaned_text)
    
    #remove the urls 
    text_frame.remove_urls()
    #remove any emails if exist
    text_frame.remove_emails()
    #remove the numbers
    #text_frame.remove_phone_numbers()

    #remove the stop words like “the”, “is” and “and”,
    #text_frame.remove_stopwords()
    #remove the punctiations
    text_frame.remove_puncts(most_common=False)
    #replacing - with space
    text_frame.text = text_frame.text.replace("-", " ")
    #replacing camelCase with space
    text_frame.text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text_frame.text)

    #remove the special chars
    text_frame.text = re.sub('[^A-Za-z]+', ' ', text_frame.text)
    text_frame.remove_special_characters()

    #replacing unwanted spaces
    text_frame.text = text_frame.text.strip()
    text_frame.text = re.sub(r'\s+', ' ', text_frame.text.strip())
    text_frame.text = text_frame.text.replace('\n', ', ')
    
    return text_frame.text.lower()
