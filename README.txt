This is readme file for question 5

How to run:
1. Install Pythonv3.6.4
2. Run the implementation by entering command on your command line python by command -
   python filename
3.The program asks
The path of training data - Enter Path of Training Data :
The path of test data -Enter Path of Testing Data :
The path where you want to store output file -Enter Path of Output File :

4. The output is stored in predictions.txt where each line has the message and its prediction separated by commas

5. I have tried different activation functions, taking unigrams and bigrams,different solver, different number of hidden layers in layer 1. To get the best model. The predictions from best model is only listed.
6. Used F1 score in K folding(10 here).
7. Also calculated Precision and Recall
8. Also calculated Confusion Matrix
9. Some of the good optimization parameters are listed in Parameter_Optimixed.pdf

These are the details of best model
Activation Function:  tanh and No of Hidden Layer in 1st layer: 100 
F1 Score: 0.9664005150528986 
Precision Score: 0.991 
Recall Score: 0.943612596553773 
Confusion matrix: 
[[800   2] 
 [ 13 208]] 

Note-
All of these libraries are imported make sure they are in the system and correctly installed
from sklearn.feature_extraction.text import CountVectorizer
from pandas import DataFrame
import numpy
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, f1_score,precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer

 
