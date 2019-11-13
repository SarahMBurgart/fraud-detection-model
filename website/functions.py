import pandas as pd
import numpy as np
import nltk
import string
import unicodedata
import re
import pickle
from collections import Counter
from nltk.probability import FreqDist
from nltk.stem import PorterStemmer
porter_stemmer=PorterStemmer()
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import RegexpTokenizer, sent_tokenize, word_tokenize
from nltk.probability import FreqDist
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn.decomposition import NMF
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
import functions as f
from sklearn.linear_model import LogisticRegression


# function to do first cleaning of text before can do tokenization
def scrub_words(text):
    """Basic cleaning of texts."""
    
    # remove html markup
    #text=re.sub("(<div style.*?>)","",text)
    #text=re.sub("(<div class.*?>)","",text)
    text=re.sub("(<.*?>)","",text)
    text=re.sub("rsquo","",text)
    text=re.sub("nbsp","",text)
    text=re.sub("ndash","",text)

    #remove non-ascii and digits
    text=re.sub("(\\W|\\d)"," ",text)
    
    #remove whitespace
    text=text.strip()
    return text

def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def load_clean():

    data = pd.read_json("data.json")

    # create fraud column as target

    data["fraud"] = data["acct_type"].str.contains("fraud")
    data.fraud.replace({True: "fraud", False: "not fraud"}, inplace=True)
    data.listed.replace({"y":1, "n": 0}, inplace=True)

    #if time - rewrite as a function to deal with any NaNs or Nulls

    fb = data.org_facebook.mean()
    tw = data.org_twitter.mean()
    data.delivery_method.fillna(.44, inplace=True)
    data.org_facebook.fillna(fb, inplace=True)
    data.org_twitter.fillna(tw, inplace=True)
    data.payout_type.replace({"ACH": 0, "CHECK": 1, "": 2}, inplace=True)
    data["num_previous_payouts"] = data["previous_payouts"].apply(lambda x: len(x))
    sd = data["sale_duration"].mean()
    data.sale_duration.fillna(sd, inplace=True)
    return data

# takes description and gives back a coeff to put in column, use with lambda 
def words_coeff(list_words):
    # bring in dictionary
    with open('/Users/sarahburgart/galvanize/week10/fraud-detection-case-study/data/words_coeffs_dict', 'rb') as sweet:
        wcd = pickle.load(sweet)
    total = 0
    # count number of each word
    l = list_words.split(",")
    fd = FreqDist(l)
    for k,v in fd.items():
        k = k.strip()
        if k in wcd.keys():
            total += (wcd[k] * v)
        else:
            pass
    return total   

# gets info from dict(s) in ticket_type columns, returns three new columns
# commented out sections reflect where there was info in the training set and not in the real life data
def ticket_types_df(data):
    tt_col = data.ticket_types
    num_tiers = []
    tickets_available = []
    tickets_sold = []
    average_ticket_price = []
    monetary_amt_sold = []
    costs = []
    counter = Counter() 
    for  d in tt_col: 
        counter.update(d)
        
    res = dict(counter)

    
    if 'quantity_total' in res:
        tickets_available.append(res['quantity_total'])
    else:
        tickets_available.append(0)
    #if "quantity_sold" in res:
        #tickets_sold.append(res["quantity_sold"])
    #else:
        #tickets_sold.append(0)
    if "cost" in res:
        average_ticket_price.append(res["cost"] / len(tt_col))
    else:
        average_ticket_price.append(0)
    #monetary_amt_sold.append(sum(costs))

    data["num_tiers"] = len(tt_col)
    data["tickets_available"] = res['quantity_total']
    #data["tickets_sold"] = tickets_sold 
    data["average_ticket_price"] = (res["cost"] / len(tt_col))
    #data["monetary_amt_sold"] = monetary_amt_sold
    
    
def previous_payouts(pp_col_value):
    amounts = []
  
    
    for lst in pp_col_value:
        if len(lst) == 0:
            data["num_previous_payouts"] = 0
            data["avg_previous_payout"] = 0
        else:   
            payouts = []
            counter = Counter()  
            for d in lst: 
                counter.update(d)
            res = dict(counter)


            if "amount" in res:
                amounts.append(res["amount"])
            else:
                amounts.append(0)


            data["num_previous_payouts"] = len(lst)
            data["avg_previous_payout"] = (sum(amounts)/len(lst))

def fit_predict_proba_model(data):
            
    first_y = data["fraud"]
    first_X = data[["body_length", "channels","delivery_method","fb_published",  
               "has_logo", "listed", "name_length",  "object_id", "org_facebook", 
               "org_twitter", "payout_type", "sale_duration", "show_map", "user_created", 
               "num_tiers", "tickets_available", "average_ticket_price", "user_age", "coeffs"]]
    # if you add in 'user_type' precision gets slightly better, recall slightly worse



    rfc = RandomForestClassifier(n_estimators=100, max_depth=12, oob_score=True, random_state=47, 
                             class_weight={"not fraud": 1, "fraud":10})

    rfc.fit(first_X, first_y)
    #ypp = rfc.predict_proba(fX_test)
    return rfc

# app.py version (the model_X.py version is currently written to take the whole df)
def to_lemma(text):
    if type(text) != str:
        text = str(text)
    scrubbed = f.scrub_words(text)
    # word tokenize
    tokenized_word = word_tokenize(scrubbed)
    # lower case the words for better frequency, tf-idf
    tokens_lower = [word.lower() for word in tokenized_word]
    # remove stopwords
    stop_words=set(stopwords.words("english"))
    filtered_desc = []
    for w in tokens_lower:
        if w not in stop_words:
            filtered_desc.append(w)
    # for lemmatization, need to pass part of speech
    pos = nltk.pos_tag(filtered_desc)
    # lemmatization with pos 
    lem = WordNetLemmatizer()
    lemmed = ""
    for w,p in pos:
        p_new = f.get_wordnet_pos(p)
        lemmed += f" {(lem.lemmatize(w,p_new))},"
    return lemmed_words


def add_new_columns(df):
    # make column for whether 'description' includes a url 
    df['desc_has_link'] = df.description.str.contains('href')

    # make column for whether 'org_desc' includes a url 
    df['org_has_link'] = df.org_desc.str.contains('href')

    # make column for email suffix code
    df['email_suffix'] = df.email_domain.str.strip().str.lower().str.extract(r'([^.]+$)')
    top9 = list(data.email_suffix.value_counts().index[:9])
    email_suffix_list = list(df.email_suffix.value_counts().index)
    email_suffix_dict = {}
    for idx in range(len(email_suffix_list)):
        if email_suffix_list[idx] in top9:
            email_suffix_dict[email_suffix_list[idx]] = idx
        else:
            email_suffix_dict[email_suffix_list[idx]] = 9
    df['email_suffix_code'] = df.copy().email_suffix.apply(lambda x: email_suffix_dict[x])
    
    # make country_match_code
    df['country_match'] = df.country == df.venue_country

    # make length of text columns
    ### Create column for the length of entry in the organization description (org_desc) field. 
    df['org_desc_len'] = df.org_desc.fillna('').str.len()

    ### Create column for the length of entry in the venue_name field. 
    df['venue_name_len'] = df['venue_name'].fillna('').str.len()

    ### Create column for the length of entry in the org_name field. 
    df['org_name_len'] = df['org_name'].fillna('').str.len()

    ### Create column for the length of entry in the payee_name field. 
    df['org_name_len'] = df['org_name'].fillna('').str.len()
