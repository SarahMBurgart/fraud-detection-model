'''
To Do:
Write a "how to use" for git repo & another for site, in meantime, notes here:
* need to start docker container with mongodb database (Zeusbot)- put it on AWS and leave it there
* currently it pulls when you trigger "score" IT WILL pull automatically and SHOW anything tagged as fraud for approval/ not
* website currently doesn't display a choice to look at the full DB, but it will!
* if all that have been tagged as fraud have been dealt with, there won't be any to deal with - then We Need another message
* choose how to display the events tagged as fraud (how much information?)
*** update the next section: to run: // there are bits in there that are checking on the db and not needed to run site
* use those bits to write up a maintenanace doc
'''

# to run:

### to check on zeusbot
'''
docker run --name mongoserver -p 27017:27017 -v "$PWD":/home/data -d mongo
docker start mongoserver
docker exec -it mongoserver mongo

### back to terminal where you ran mongo

show dbs
use zeusbot
db.zeus_files

zb = client.zeusbot
collection = zb.zeus_files
collection.insert_one({'name':'Zeus', 'city':'Olympus'})
'''
### now it is created, you have some options
# db.zeus_files.find_one({'field': 'entry'})
# db.zeus_files.find_one()
# db.zeus_files.find()
# db.zeus_files.estimated_document_count()



# code for flask, python code for a site that will load a model and data from a stream upon loading and create probs
# the probs will show when you follow the score link
# EDA currently has a couple graphs with text
# report has info on our process

from flask import Flask, request
import pandas as pd
import dill as pickle
import pymongo
import numpy as np
from collections import Counter
import functions as f

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return ''' <p> Speak, friend, and enter:
                   <a href="/report">report</a> 
                   <a href="/EDA">EDA</a>
                   <a href="/score">score</a></p> '''

@app.route('/report', methods=['GET'])
def report():
    # load report.md in here and format so it is pretty
    return ''' <h1> A quick report!</h1> 
    <pre>
An overview of a chosen "optimal" modeling technique, with:


process flow:

we worked together to do basic EDA to get an understanding of the data.

we then split tasks, regularly checking in to see if we needed to switch tasks based on what the other person was doing or based on the time remaining for the project.

a key factor was testing the code and principles of our project as we went along. 


preprocessing:

because we did EDA on the feature importances both through a basic Random Forest Classifier and through plots, we were able to save time and energy by not doing pre-processing on features which were not essential to our model.

a preprocessing factor that came to our attention after a few iterations was the discrepancy between the data in our training set and the set we will be retrieving through the API. 

accuracy metrics selected:

While we did track oob score accuracy of our Random Forest Classifier, we decided precision and recall would be the most pertinent metrics and between those two favored lowering the false negatives.


validation and testing methodology:

80/20 train test split

critical thinking: 100% recall brought into question leakage ... and we were right!


parameter tuning involved in generating the model:

tried many different levels of class weights, estimators, and max_depth. 

further steps you might have taken if you were to continue the project.

better website interface

use of time and words 




</pre>'''

@app.route('/EDA', methods=['GET'])

def EDA():
    return ''' <h1> Initial Data Exploration </h1>
                <h3> Our data began with 45 columns, so our EDA focused on which 
               which features seemed to have a correlation with instances of fraud.
                Please forgive the y-axis labels, 1 = Fraud and 2 = Not Fraud </h3> 
               <img src="static/boxplot2.png" alt="Girl in a jacket">
               <h3> We focused on several variables, and then learned that the variables are not included in the live data, so some of our EDA became irrelevant. </h3>
               <img src="static/leaks.png" alt="Girl in a jacket">
               <h3> Long & Lat were irrelevant, but the plots look nice </h3>
               <img src="static/scatter.png" alt="Girl in a jacket">
               <h3> Look! A Precision-Recall Curve </h3>
               <img src="static/prc.png" alt="Girl in a jacket">'''


def words_df(df):
    # takes df, returns df with coeffs column
    # remove nonsense
    text_col = df.description
    # returned list of strings of lemmatized words
    lemmed = text_col.apply(lambda x: f.to_lemma(x))
    lemmed.fillna("", inplace=True)
    lemmed.replace(" nan,", "", inplace=True)
    lem = list(lemmed)
    # list of string of strings needed for next step
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
    df["coeffs"] = map(f.words_coeff, new_lemmed)
    return df

def load_clean(data):
    # char to int
    data.listed.replace({"y":1, "n": 0}, inplace=True)
    data.payout_type.replace({"ACH": 0, "CHECK": 1, "": 2}, inplace=True)
    
    # get info from dict
    f.ticket_types_df(data)
    
    # if time - rewrite as a function to deal with any NaNs or Nulls
    values = {'org_facebook': 7.97, 'org_twitter': 4.31,"sale_duration": 46.87, 'delivery_method': .44}
    data.fillna(value=values, inplace=True)
    
    # nltk
    words_df(data)

    return data


@app.route('/score', methods=['GET'])
def score():
    
    '''for data in df:
        X = data[["body_length", "channels","delivery_method","fb_published",  
                   "has_logo", "listed", "name_length",  "object_id", "org_facebook",
                   "org_twitter", "payout_type", "sale_duration", "show_map", "user_created", 
                   "num_tiers", "tickets_available", "average_ticket_price", "user_age", coeffs]]

        # and put through model
        probability = model.predict_proba(X)
        probability =  round(probability[:,0], 4)'''

        # add row to database
        # add other keys and values as needed


        # maybe send to next function

    return f''' probability of fraud: {probs}  ''' 

# options to dismiss or submit fraudulent data report
# @app.route('/submit', methods=['GET'])
# def home():
#   return ''' <p> nothing here, friend, but a link to 
#                   <a href="/hello">hello</a> and an 
#                   <a href="/form_example">example form</a> </p> '''


if __name__ == '__main__':
    
    
    # add unpickled model
    with open('modelX.pkl', 'rb') as modelZ:
        model = pickle.load(modelZ)

    # add connect to database
    # EC2 instance = ec2-18-218-83-33.us-east-2.compute.amazonaws.com
    connection_string = 'mongodb://localhost:27017/'


    client = pymongo.MongoClient(connection_string)
    zb = client.zeusbot
    Zeusbot = zb.zeus_files
    
    ZB = list(Zeusbot.find())
    print(ZB)
    probs = []


    print("hello!")
    for bot in ZB:
        bot['previous_payouts'] = 0
        datum = pd.DataFrame.from_dict(bot)
        datum = load_clean(datum)
        

    
        X = datum[["body_length", "channels","delivery_method","fb_published",  
                   "has_logo", "listed", "name_length",  "object_id", "org_facebook",
                   "org_twitter", "payout_type", "sale_duration", "show_map", "user_created", 
                   "num_tiers", "tickets_available", "average_ticket_price", "user_age"]]

        # and put through model
        probability = model.predict_proba(X)
        probability =  np.round(probability[0], 4)
        probs.append(probability)
        # add row to database
        # add other keys and values as needed


        # maybe send to next function

    # return f''' probability of fraud: {probability}  ''' #.format(reversed_string)

    
   
    app.run(host='0.0.0.0', port=8087, debug=True)
    
    

    
    
    
    