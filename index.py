import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import accuracy_score

(train_X, train_y), (test_X, test_y) = mnist.load_data()

SIZE = train_X.shape[1]   

train_X = train_X.reshape(-1, SIZE*SIZE)
test_X = test_X.reshape(-1, SIZE*SIZE)

train_X = train_X / 255
test_X = test_X / 255

class Layer:
    def __init__(self, neuron_number, weight_number):
        self.weights = np.random.random((neuron_number, weight_number)) * 5 - 2
        self.b = np.random.random(neuron_number) * 5 - 2
    
    def sigmoid(self, s, alpha=0.1):
        return 1 / (1 + np.exp(-alpha*s))
    
    def change_weight(self, subtracted_weight):
        self.weights -= subtracted_weight
    
    def change_b(self, subtracted_b):
        self.b -= subtracted_b
    
    def layerPass(self, input_data):
        s = np.dot(self.weights, input_data) + self.b
        return self.sigmoid(s)
    
    def layerOutPuts(self, data): 
        length = self.weights.shape[0]
        self.outputs = np.zeros(length)
        self.outputs = self.layerPass(data)
class Network:
    def __init__(self, network_data):
        self.layerList = []
        for i in range(1, len(network_data)):
            neuron_number = network_data[i]
            weight_number = network_data[i - 1]
            self.layerList.append(Layer(neuron_number, weight_number))
        self.references = np.eye(network_data[-1])
            
    def deriv_sigmoid(self, y):
        return y * (1 - y)

    def fit(self, input_datas, y_trues, learn_rate=0.1, epochs=1):
        for epoch in range(epochs):
            for input_data, y_true in zip(input_datas, y_trues):
                y_true = np.array(self.references[y_true])
                
                for i in range(len(self.layerList)):
                    
                    if i == 0:
                        self.layerList[i].layerOutPuts(input_data)
                    
                    else:
                        self.layerList[i].layerOutPuts(self.layerList[i-1].outputs)
                
                y_pred = self.layerList[-1].outputs         
                d_L_d_ypred = y_pred - y_true
                
                for i in range(len(self.layerList) - 1, -1, -1):
                    
                    if i == len(self.layerList) - 1:
            
                        subtracted_b = d_L_d_ypred * self.deriv_sigmoid(y_pred) 
                        subtracted_weight = np.dot(subtracted_b.reshape(-1, 1), self.layerList[i-1].outputs.reshape(1, -1))
                        self.layerList[i].change_weight(learn_rate * subtracted_weight)
                        self.layerList[i].change_b(learn_rate * subtracted_b)
                    
                    elif i != 0:
                           
                        next_layer_weights = np.array(self.layerList[i+1].weights)   
                        product = np.dot(subtracted_b, next_layer_weights)
                        subtracted_b = self.deriv_sigmoid(self.layerList[i].outputs) * product
                        subtracted_weight = np.dot(subtracted_b.reshape(-1, 1), self.layerList[i-1].outputs.reshape(1, -1))
                        self.layerList[i].change_weight(learn_rate * subtracted_weight)
                        self.layerList[i].change_b(learn_rate * subtracted_b)
                    
                    else:
                                                        
                        next_layer_weights = np.array(self.layerList[i+1].weights)
                        product = np.dot(subtracted_b, next_layer_weights)
                        subtracted_b = self.deriv_sigmoid(self.layerList[i].outputs) * product
                        subtracted_weight = np.dot(subtracted_b.reshape(-1, 1), input_data.reshape(1, -1))
                        self.layerList[i].change_weight(learn_rate * subtracted_weight)
                        self.layerList[i].change_b(learn_rate * subtracted_b)
                
    def predictProba(self, input_datas):
        network_outputs = []
        for n in range(input_datas.shape[0]):
            for i in range(len(self.layerList)):
                if i == 0:
                    self.layerList[i].layerOutPuts(input_datas[n])
                else:
                    self.layerList[i].layerOutPuts(self.layerList[i-1].outputs)
            network_outputs = np.append(network_outputs, self.layerList[len(self.layerList) - 1].outputs)
        return network_outputs.reshape(input_datas.shape[0], 10)
    
    def predict(self, input_datas):
        network_outputs = self.predictProba(input_datas)
        predicts = []
        for i in range(network_outputs.shape[0]):
            predicts = np.append(predicts, np.argmax(network_outputs[i]))
        return predicts
      
def graphDrawing(test, train, label1, label2, label_x, label_y, location):
    x = range(len(test))
    plt.grid(True)
    plt.plot(x, test, label = label1)
    plt.plot(x, train, label = label2)
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.legend(loc = location)
  
nt = Network([784, 32, 16, 10])

train_loss = test_loss = test_accuracy = train_accuracy = []
test_indices = np.random.choice(test_X.shape[0], 100)
train_indices = np.random.choice(train_X.shape[0], 100)

train_loss = np.append(train_loss, ((np.array(nt.references[train_y[train_indices]]) - nt.predictProba(train_X[train_indices]))**2).mean())
test_loss = np.append(test_loss, ((np.array(nt.references[test_y[test_indices]]) - nt.predictProba(test_X[test_indices]))**2).mean())
train_accuracy = np.append(train_accuracy, accuracy_score(nt.predict(train_X[train_indices]), train_y[train_indices]))
test_accuracy = np.append(test_accuracy, accuracy_score(nt.predict(test_X[test_indices]), test_y[test_indices]))

examples = 60000
for i in range(10):
    nt.fit(train_X[:examples], train_y[:examples], 0.05, 1)
    
    test_indices = np.random.choice(test_X.shape[0], 100)
    train_indices = np.random.choice(train_X.shape[0], 100)
    train_loss = np.append(train_loss, ((np.array(nt.references[train_y[train_indices]]) - nt.predictProba(train_X[train_indices]))**2).mean())
    test_loss = np.append(test_loss, ((np.array(nt.references[test_y[test_indices]]) - nt.predictProba(test_X[test_indices]))**2).mean())
    train_accuracy = np.append(train_accuracy, accuracy_score(nt.predict(train_X[train_indices]), train_y[train_indices]))
    test_accuracy = np.append(test_accuracy, accuracy_score(nt.predict(test_X[test_indices]), test_y[test_indices])) 
  
graphDrawing(train_loss, test_loss , 'Train Loss', 'Test Loss', 'Epoch', 'Accuracy', 'upper right')

graphDrawing(train_accuracy, test_accuracy, 'Train Accuracy', 'Test Accuracy', 'Epoch', 'Accuracy', 'lower right') 
