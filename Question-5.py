#all imports

from sklearn.feature_extraction.text import CountVectorizer
from pandas import DataFrame
import numpy
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, f1_score,precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer

#defining String Constants
HAM = 'ham'
SPAM = 'spam'

#Function for building data frame from data
def build_data_frame(body,classification,x):
    rows = []
    index = []
    rows.append({'text': body[x], 'class': classification[x]})
    index.append(x)
    data_frame = DataFrame(rows, index=index)
    return data_frame

#taking input all paths
train_data_path= input("Enter Path of Training Data :")
test_data_path= input("Enter Path of Testing Data :")
output_path= input("Enter Path of Output File :")


classification =[]
with open(train_data_path,"r",encoding="ISO-8859-1") as f:
    #reading files one by one
	lines=f.readlines()
	for n in range(len(lines)):
			#checking if there is no empty line
			if not len(lines[n].strip()) == 0:
				if(lines[n][-6:] == ",spam\n"):
					lines[n] = lines[n][:-5]
					classification.append("spam")
				elif(lines[n][-5:] == ",ham\n"):
					lines[n] = lines[n][:-4]
					classification.append("ham")

#building data frames or converting them to frames
data = DataFrame({'text': [], 'class': []})
for n in range(len(lines)):
	data = data.append(build_data_frame(lines,classification,n))
data = data.reindex(numpy.random.permutation(data.index))

#building classifier and setting best parameters
classifier = MLPClassifier(activation = 'tanh', solver='lbfgs', alpha=1e-5,learning_rate ='adaptive',hidden_layer_sizes=(100, 10), random_state=1)

#constructing pipeline
pipeline = Pipeline([
    ('vectorizer',  CountVectorizer()),
    ('classifier',  classifier ) ])

#opening the files
with open(test_data_path,"r",encoding="ISO-8859-1") as f:
    #reading files one by one
	test_lines=f.readlines()
	for n in range(len(test_lines)):
		#checking if there is no empty line
		if not len(test_lines[n].strip()) == 0:
			test_lines[n] = test_lines[n][:-1]

#intializing
k_fold = KFold(n_splits=10)
scores = []
p_scores = []
r_scores = []
confusion = numpy.array([[0, 0], [0, 0]])

#k folding
for train_indices, test_indices in k_fold.split(data):
    #training data by k folding
    train_text = data.iloc[train_indices]['text'].values
    train_y = data.iloc[train_indices]['class'].values

    #testing data by k folding
    test_text = data.iloc[test_indices]['text'].values
    test_y = data.iloc[test_indices]['class'].values

    #pipelining
    pipeline.fit(train_text, train_y)

    #predicting on test data of k fold
    predictions = pipeline.predict(test_text)

    #generating measures
    confusion += confusion_matrix(test_y, predictions)
    score = f1_score(test_y, predictions, pos_label=SPAM)
    p_score = precision_score(test_y, predictions, pos_label=SPAM)
    r_score = recall_score(test_y, predictions, pos_label=SPAM)
    scores.append(score)
    p_scores.append(p_score)
    r_scores.append(r_score)

#printing the results
print('Total emails classified:', len(data))
print('F1 Score:', sum(scores)/len(scores))
print('Precision Score:', sum(p_scores)/len(p_scores))
print('Recall Score:', sum(r_scores)/len(r_scores))
print('Confusion matrix:')
print(confusion)

#running the model on test data
predictions = pipeline.predict(test_lines)

#writing it to the file
with open(output_path+"/Predictions.txt","w+",encoding="ISO-8859-1") as fu:
    for x in range(len(test_lines)):
        fu.write(test_lines[x] + " , " + predictions[x]+"\n")
