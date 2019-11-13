#from model_X import modelX
import pandas as pd
import dill as pickle
from scipy.stats import bernoulli 
import random

#test = pd.read_csv('test_script_examples.csv')
#loaded_model = pickle.load(open('modelX.pkl', 'rb'))
def unpickle(model)
with open('modelX.pkl', 'rb') as modelZ:
    model = pickle.load(modelZ)

# result = loaded_model(test[0])
# print(result)
test = pd.read_csv('test_script_examples.csv')
outcome, probability = model(test.iloc[0,:])
print(outcome, round(probability, 2))

