'''
Python Version 3.6.5
Libraries Used : Pandas
'''

import pandas as pd

'''
Loading the train and test data into matrices
'''
trainSet = pd.read_csv('trainSeeds.csv', header=None)
trainSet = trainSet.values
testSet = pd.read_csv('testSeeds.csv', header=None)
testSet = testSet.values

'''
The designed neural network consists of 3 Precptrons, one for each Class. The network uses
simple feedback algorithm to update its weights and learning. The network consists of 5 functions.
'''
def Perceptron(traindata, testdata, learning_Rate, No_iterations, Class):
    predictions = list()
    weights =  Perceptron_train(traindata, learning_Rate, No_iterations, Class)
    for X in testdata:
        prediction = Perceptron_predict(X, weights)
        predictions.append(prediction)
    return [predictions,weights]

'''
The Perceptron_train function has the same parameters of the Perceptron function coded above
except the testdata. The weights are initialized and logged to a csv file.The error is 0 if
predicted output is correct if not its + or - 1.
'''
def Perceptron_train(traindata, lr, itr, Class):
    weights = [[0.0 for i in range (len(traindata[0]))] for j in range(Class)]
    iW = pd.DataFrame(weights).transpose()
    iW.to_csv ('Initial_Weights.csv',header=True, index=True)
    '''
    for loop over the Number of iterations, all observations and wheat classes respectively
    '''
    for m in range(itr):
        for X in traindata:
            prediction = Perceptron_predict(X, weights)
            for a in range(Class):
                if(prediction[a] == 1 and X[-1] == (a+1)):error = 0.0
                elif(prediction[a] == 0 and X[-1] != (a+1)):error = 0.0
                elif(prediction[a] == 0 and X[-1] == (a+1)):error = 1.0
                else:error = -1.0
                '''
                updating weights in simple feedback learning
                '''
                weights[a][0] = weights[a][0] + lr*error
                for i in range(len(X)-1):
                    weights[a][i+1] = weights[a][i+1] + lr*error*X[i]
    return weights

 
'''
The Predict function maps the inputs to the outputs depending on the activation
rule.
'''
def Perceptron_predict(X, weights):
    al = list()
    for a in range(len(weights)):
        activation = weights[a][0]
        for i in range(len(X)-1):
            activation += weights[a][i+1] * X[i]
        al.append(activation)
        '''
   The class with the most activation is predicted [1,0,0] for Kama [0,1,0] for
   Rosa and [0,0,1] for Canadian.
        '''
    activationrule = max(al[0],al[1],al[2])
    if activationrule == al[0]:return [1,0,0]
    elif activationrule == al[1]:return [0,1,0]
    elif activationrule == al[2]:return [0,0,1]
    else : return [0,0,0]

'''
Function for calculating and logging the confusion matrix, precision and 
recall of the network to a csv file.
'''
def Classification_report(actual, predicted):
    for i in range(len(actual)):
        c[int(actual[i]-1)][int(predicted[i]-1)] += 1
    correct = c[0][0]+c[1][1]+c[2][2]  
    '''
    Precision
    '''
    precision[0] = c[0][0]/(c[0][0]+c[1][0]+c[2][0])
    precision[1] = c[1][1]/(c[1][1]+c[0][1]+c[2][1])
    precision[2] = c[2][2]/(c[2][2]+c[1][2]+c[0][2])
    '''
    Recall
    '''
    recall[0] = c[0][0]/(c[0][0]+c[0][1]+c[0][2])
    recall[1] = c[1][1]/(c[1][1]+c[1][0]+c[1][2])
    recall[2] = c[2][2]/(c[2][2]+c[2][1]+c[2][0])
    
    cm = pd.DataFrame(c)
    cm.columns=['Kama', 'Rosa','Canadian']
    ps = pd.DataFrame(precision).transpose()
    rc = pd.DataFrame(recall).transpose()
    ps.to_csv('Precision.csv', header=True, index=True)
    rc.to_csv('Recall.csv', header=True, index=True)
    cm.to_csv('ConfusionMatrix.csv', header=True, index=True)
    return (correct / float(len(actual))*100.0)

'''
The Preceptron_fit function fits the data into the model it takes all the parameters 
listed below.
'''
def Perceptron_fit(dataset,testdata,model,learning_Rate,No_iterations,Class):
    results = list()
    train_set = dataset
    test_set = testdata
    [predicted,weights] = model(train_set,test_set,learning_Rate,No_iterations,Class)
    predictedf = list()
    for i in range(len(predicted)):
        for j in range(len(predicted[0])):
            if(predicted[i][j] == 1):
                predictedf.append(j+1)
    actual = [X[-1] for X in test_set]
    weightVals = pd.DataFrame(weights).transpose()
    weightVals.to_csv('Final_Weights.csv', header=True, index=True)
    accuracy = Classification_report(actual, predictedf)
    results.append(accuracy)
    return results


learning_Rate = 0.1       
No_iterations = 550   
Class = 3
c = [[0.0 for i in range(Class)] for j in range(Class)] 
precision = [0.0 for i in range(Class)]
recall = [0.0 for i in range(Class)]

'''
Run the the below statement to excute the entire networking after compiling all
of the above lines of code.
'''
results = Perceptron_fit(trainSet,testSet,Perceptron,learning_Rate,No_iterations,Class)


print("Number of Iterations: %d" % (No_iterations))
print("Accuracy: %.3f%%" % (sum(results)/float(len(results))))
print("Learning Rate: %s" % (learning_Rate))


##Best results with 550 epochs




      
      
   
   
   
   
   
   