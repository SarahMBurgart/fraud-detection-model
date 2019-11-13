# fix to actual html, and update with more thoughts!!! (ha)

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




</pre>