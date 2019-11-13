# this code makes a model with the data from data.json and pickles it

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import dill as pickle
import re
from nltk.tokenize import RegexpTokenizer, sent_tokenize, word_tokenize
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.probability import FreqDist
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn.linear_model import LogisticRegression
import functions as f

def load_clean():
    # load data
    data = pd.read_json("/Users/sarahburgart/galvanize/week10/fraud-detection-case-study/data/data.json")

    # create fraud column as target
    data["fraud"] = data["acct_type"].str.contains("fraud")
    #data.fraud.replace({True: 1, False: 0}, inplace=True)
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


# why isn't included in the model?
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
    return lemmed

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

def words(text_col):
    # takes text, returns lemmatized words
    # remove nonsense
    lemmed = text_col.apply(lambda x: to_lemma(x))
    lemmed.fillna("", inplace=True)
    lemmed.replace(" nan,", "", inplace=True)
    lem = list(lemmed)
    new_lemmed = []
    for string in lem:
        new_str = ''
        s = string.split(",")
        for word in s:
            if len(word) > 3:
                new_str += f"{word},"
            else:
                pass
        new_lemmed.append(new_str)
    # vectorize
    tf = TfidfVectorizer(stop_words='english', analyzer = "word", max_features = 10000)
    text_tf = tf.fit_transform(new_lemmed)
    # factorize
    model = NMF(n_components=20, init='random', random_state=42)
    W = model.fit_transform(text_tf.T)
    H = model.components_
    descriptions = H.T
    des = pd.DataFrame(descriptions)
    target = pd.read_csv('/Users/sarahburgart/galvanize/week10/fraud-detection-case-study/data/S_target.csv')
    # logistic Regression to get coefficients for factors
    lr = LogisticRegression(random_state=42, verbose=1).fit(des, target["fraud"])
    # create dictionary with coeff for each word
    features = pd.DataFrame(W)
    feats = tf.get_feature_names()
    features["names"] = feats
    features.set_index("names", inplace=True)
    coeffs = features * lr.coef_
    coeffs["total"] = coeffs.sum(axis=1)
    words_coeffs_dict = coeffs["total"].to_dict()
    # send it to data folder to use in website for new data
    with open('/Users/sarahburgart/galvanize/week10/fraud-detection-case-study/data/words_coeffs_dict', 'wb') as sweet:
        pickle.dump(words_coeffs_dict, sweet)
    # then need to get coeffs to use to train model
    coeffs = []
    for string in new_lemmed:
        total = 0
        # count number of each word
        l = string.split(",")
        fd = FreqDist(l)
        for k,v in fd.items():
            k = k.strip()
            if k in words_coeffs_dict.keys():
                total += (words_coeffs_dict[k] * v)
            else:
                pass
        coeffs.append(total)
    data["coeffs"] = coeffs  

def fit_predict_proba_model(data):           
    first_y = data["fraud"]
    first_X = data[["body_length", "channels","delivery_method","fb_published",  
               "has_logo", "listed", "name_length",  "object_id", "org_facebook", 
               "org_twitter", "payout_type", "sale_duration", "show_map", "user_created", 
               "num_tiers", "tickets_available", "average_ticket_price", "user_age", "coeffs"]]
    # if you add in 'user_type' precision gets slightly better, recall slightly worse
    rfc = RandomForestClassifier(n_estimators=100, max_depth=12, oob_score=True, random_state=47, 
                             class_weight={False: 1, True:10})
    rfc.fit(first_X, first_y)
    #ypp = rfc.predict_proba(fX_test)
    return rfc


def ticket_types_col(tt_col):
    num_tiers = []
    tickets_available = []
    average_ticket_price = []   
    for  _ in tt_col:
        costs = []
        counter = Counter()  
        for d in _: 
            counter.update(d)
            costs.append(d["cost"]*d["quantity_sold"])
        res = dict(counter)
        num_tiers.append(len(_))
        if 'quantity_total' in res:
            tickets_available.append(res['quantity_total'])
        else:
            tickets_available.append(0)    
        if "cost" in res:
            average_ticket_price.append(res["cost"] / len(_))
        else:
            average_ticket_price.append(0)

    data["num_tiers"] = num_tiers
    data["tickets_available"] = tickets_available
    data["average_ticket_price"] = average_ticket_price 

if __name__ == '__main__':
    data = load_clean()
    ticket_types_col(data.ticket_types)
    previous_payouts(data.previous_payouts)
    words(data.description)
    
    modelX = fit_predict_proba_model(data)
    
    with open('modelX2.pkl', 'wb') as my_pickle:
        pickle.dump(modelX, my_pickle)

    



