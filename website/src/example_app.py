from flask import Flask, request
import predict
app = Flask(__name__)


@app.route('/', methods=['GET'])
def home():
    return ''' <p> Welcome to the fraud detection services of Dogtective Lenny Briscoe
                   <a href="/hello">hello</a> and an 
                   <a href="/form_example">example form</a>... Also, get some fraud predictions
                   <a href="/score">here</a> </p> '''

@app.route('/hello', methods=['GET'])
def hello_world():
    return ''' <h1> Hello, World!</h1> '''

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

@app.route('/score', methods=['POST'])
def score():
    text = str(request.form['some_string'])
    reversed_string = text[-1::-1]
    return ''' output: {}  '''.format(reversed_string)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)