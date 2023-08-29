import numpy
import scipy.special
import torch

class neuralnetwork:
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        self.inodes=inputnodes
        self.hnodes=hiddennodes
        self.onodes=outputnodes
        self.wih=numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
        self.who=numpy.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))
        self.lr=learningrate
        self.activation_function=lambda x:scipy.special.expit(x) 
        pass
    def train(self,inputs_list,targets_list):
        inputs=numpy.array(inputs_list,ndmin=2).T
        targets=numpy.array(targets_list,ndmin=2).T
        hidden_inputs=numpy.dot(self.wih,inputs)
        hidden_outputs=self.activation_function(hidden_inputs)
        finial_inputs=numpy.dot(self.who,hidden_outputs)
        finial_outputs=self.activation_function(finial_inputs)
        output_errors = targets - finial_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors) 
        self.who += self.lr * numpy.dot((output_errors * finial_outputs * (1.0 - finial_outputs)), numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))
        pass
    def query(self,inputs_list):
        inputs=numpy.array(inputs_list,ndmin=2).T
        hidden_inputs=numpy.dot(self.wih,inputs)
        hidden_outputs=self.activation_function(hidden_inputs)
        finial_inputs=numpy.dot(self.who,hidden_outputs)
        finial_outputs=self.activation_function(finial_inputs)
        print(finial_outputs)
        return finial_outputs
        
input_nodes=3
hidden_nodes=3
output_nodes=3
learning_rate=0.5
n=neuralnetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)

n.query([1.0,0.5,-1.5])

    