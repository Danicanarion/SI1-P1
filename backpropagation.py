import numpy as np
import sys
import time
from inputlayer import InputLayer
from hiddenlayer import HiddenLayer
from outputlayer import OutputLayer
import random
import json

def sig(x):
   return 1/(1+np.exp(-x.astype(float)))


def sigDx(x): 
    y = sig(x)
    return np.multiply(y,(1-y))


l2_cost = (lambda Yp, Yr: np.mean(np.power((Yp-Yr), 2)),
           lambda Yp, Yr: (Yp-Yr))


class BackPropagation(object):
    """Class BackPropagation:

       Attributes:
         eta.- Learning rate
         number_iterations.-
         ramdon_state.- Random process seed
         input_layer_.-
         hidden_layers_.-
         output_layer_.-
         sse_while_fit_.-

       Methods:
         __init__(p_eta=0.01, p_iterations_number=50, p_ramdon_state=1)
         fit(p_X_training, p_Y_training, p_X_validation, p_Y_validation,
             p_number_hidden_layers=1, p_number_neurons_hidden_layers=numpy.array([1]))
         predict(p_x) .- Method to predict the output, y

    """
    wights = None
    model = None

    def __init__(self, p_eta=0.5, p_number_iterations=20, p_random_state=None):
        self.eta = p_eta
        self.number_iterations = p_number_iterations
        self.random_seed = np.random.RandomState(p_random_state)

    def fit(self, p_X_training,
            p_Y_training,
            p_X_validation,
            p_Y_validation,
            p_number_hidden_layers=1,
            p_number_neurons_hidden_layers=np.array([1])):
        
        count = np.bincount(p_Y_training.flatten())
        alfa = count[1] / count[0]
        (m, n) = p_X_training.shape
        
        self.input_layer_ = InputLayer(p_X_training.shape[1])
        self.hidden_layers_ = []
        for v_layer in range(p_number_hidden_layers):
            if v_layer == 0:
                self.hidden_layers_.append(HiddenLayer(p_number_neurons_hidden_layers[v_layer],
                                                       self.input_layer_.number_neurons,
                                                       sig))
            else:
                self.hidden_layers_.append(HiddenLayer(p_number_neurons_hidden_layers[v_layer],
                                                       p_number_neurons_hidden_layers[v_layer - 1],
                                                       sig))
        self.output_layer_ = OutputLayer(p_Y_training.shape[1],
                                         self.hidden_layers_[self.hidden_layers_.__len__() - 1].number_neurons, sig)

        
        self._load_weights()
       
        self.layers = self.hidden_layers_.copy()
        self.layers.append(self.output_layer_)
 
        # ...
        for iter in range(self.number_iterations):
            for i in range(m):
                out = self._forward_pass(p_X_training[i,:])   
                self._backward_pass(out, p_Y_training[i], alfa)
                
            loss = l2_cost[0](p_Y_training, self.predict(p_X_training))
            acc = self.get_accuracy(p_Y_validation, p_X_validation)
            print("acc: {}, loss: {}".format(acc, loss))

    def _backward_pass(self, out, y, alfa):
        delta = None
        _w = None
        expected, calculated = y[0], out[-1][1][0]
        if expected == 0 and random.random() > alfa:
            return
        for idx, layer in enumerate(reversed(self.layers)):
            outIndex = len(out) - idx - 1
            a = np.array(out[outIndex][1])
            z = np.array(out[outIndex][0])
            layer_input = np.array(out[outIndex - 1][1], ndmin=2)
            if idx == 0:
                delta = (expected - calculated) * sigDx(z)
            else: 
                delta = (delta @ _w.T) * sigDx(z)
           
            _w = layer.w[1:,:]
            
            delta = np.array(delta, ndmin=2)
                     
            layer.w[0,:] = layer.w[0,:] + self.eta * np.mean(delta)
            layer.w[1:,:] = layer.w[1:,:] + self.eta * layer_input.T @ delta


    def _forward_pass(self, x):
        out = [(x, x)]
        for layer in self.layers:
            z = layer._net_input(out[-1][1])
            a = layer._activation(z)
            out.append((z, a))
        
        return out

    def get_accuracy(self, p_Y_target, p_X):
        total = len(p_Y_target)
        predicted = self.predict(p_X)
        count = 0
        for c in range(total):
            if p_Y_target[c] == predicted[c]:
                count += 1

        return count / total * 100
    
    def load_weights(self, sourceFile):
        with open(sourceFile) as sf:
            self.wights = json.load(sf)

    def _load_weights(self):
        if self.wights != None:
            self._load_preload_weights()
        else:
            self._load_default_weights()

    def _load_preload_weights(self):
        self.input_layer_.w  = np.array(self.wights['input'])
        for l, w in zip(self.hidden_layers_, self.wights['hidden']):
            l.w = np.array(w)
        self.output_layer_.w  = np.array(self.wights['output'])


    def _load_default_weights(self):
        self.input_layer_.init_w(self.random_seed)
        for v_hidden_layer in self.hidden_layers_:
            v_hidden_layer.init_w(self.random_seed)
        self.output_layer_.init_w(self.random_seed)


    def load_model(self, sourceFile):
        with open(sourceFile) as sf:
            self.model = json.load(sf)

    def _load_model(self):
        if self.model != None:
            pass
        else:
            pass

    def save_model(self, targetFile):
        inputLayout = self.input_layer_.number_neurons.tolist()
        hiddenLayout = []
        for h in self.hidden_layers_:
            hiddenLayout.append((h.number_neurons.tolist(),
                                 h.number_inputs_each_neuron.tolist()))
        outputLayout = (self.output_layer_.number_neurons.tolist(),
                        self.output_layer_.number_inputs_each_neuron.tolist())
        data = {'input': inputLayout,
                'hidden': hiddenLayout,
                'output': outputLayout}
        with open(targetFile, 'w+') as tf:
            json.dump(data, tf)

    def save_weights(self, targetFile):
        data = {'input': self.input_layer_.w.tolist(),
                'hidden': [layer.w.tolist() for layer in self.hidden_layers_],
                'output': self.output_layer_.w.tolist()}
        with open(targetFile, 'w+') as tf:
            json.dump(data, tf)

    def predict(self, p_X): 
        return self.output_layer_._quantization(self.get_probability(p_X))
    
    def get_probability(self, p_X):
        v_Y_input_layer_ = self.input_layer_.predict(p_X)
        v_X_hidden_layer_ = v_Y_input_layer_
        for v_hiddenlayer in self.hidden_layers_:
            v_Y_hidden_layer_ = v_hiddenlayer.predict(v_X_hidden_layer_)
            v_X_hidden_layer_ = v_Y_hidden_layer_
        v_X_output_layer_ = self.output_layer_._net_input(v_Y_hidden_layer_)
        v_Y_output_layer_ = self.output_layer_._activation(v_X_output_layer_)
        return v_Y_output_layer_

