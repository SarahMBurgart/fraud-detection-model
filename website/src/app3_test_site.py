
from flask import Flask, request
#import predict, render_template
import os
from flask import url_for


app = Flask(__name__)


@app.route('/', methods=['GET'])
def home():
    return ''' <p> Welcome to the fraud detection services of Dogtective Lenny Briscoe
                   <a href="/hello">hello</a> and an 
                   <a href="/form_example">example form</a>... Also, get some EDA 
                   <a href="/EDA">here</a> </p>
                   '''

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
               <img src="static/prc.png" alt="Girl in a jacket">
               '''

@app.route('/form_example', methods=['GET'])
def form_display():
    return ''' <form action="/string_reverse" method="POST">
                <input type="text" name="some_string" />
                <input type="submit" />
               </form>
             '''

@app.route('/string_reverse', methods=['POST'])
def reverse_string():
    text = str(request.form['some_string'])
    reversed_string = text[-1::-1]
    return ''' output: {}  '''.format(reversed_string)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)