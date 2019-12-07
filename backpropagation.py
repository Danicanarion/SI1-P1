import numpy as np
import sys
from inputlayer import InputLayer
from hiddenlayer import HiddenLayer
from outputlayer import OutputLayer


def sig(x):
   return 1/(1+np.e**(-x))


def sigDx(x): 
    return np.multiply(x,(1-x))


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

    def __init__(self, p_eta=0.01, p_number_iterations=10, p_random_state=None):
        self.eta = p_eta
        self.number_iterations = p_number_iterations
        self.random_seed = np.random.RandomState(p_random_state)

    def fit(self, p_X_training,
            p_Y_training,
            p_X_validation,
            p_Y_validation,
            p_number_hidden_layers=1,
            p_number_neurons_hidden_layers=np.array([1])):

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

        self.input_layer_.init_w(self.random_seed)
        for v_hidden_layer in self.hidden_layers_:
            v_hidden_layer.init_w(self.random_seed)
        self.output_layer_.init_w(self.random_seed)

        # ...
        for iter in range(self.number_iterations):

            for i in range(m):
                #forward
                x = p_X_training[i,:]
                y = p_Y_training[i]
                out = [(x, x)]
                layers = self.hidden_layers_.copy()
                layers.append(self.output_layer_)
                for layer in layers:
                    net_input = []
                    for n in range(layer.number_neurons):
                        suma = layer.w[0, n]
                        for k in range(layer.number_inputs_each_neuron):
                            suma = suma + layer.w[k, n] * out[-1][1][k]
                        net_input.append(suma)
                    activation = []
                    for ni in net_input:
                        activation.append(sig(ni))
                    out.append((net_input, activation))
                
                #backward
                
                delta = None
                _w = None
                for idx, layer in enumerate(reversed(layers)):
                    outIndex = len(out) - idx - 1
                    a = np.array(out[outIndex][1])
                    layer_input = np.array(out[outIndex - 1][1], ndmin=2)
                    if idx == 0:
                        delta = (y - out[-1][1]) * sigDx(a)
                    else: 
                        delta = (delta @ _w.T) * sigDx(a)

                    _w = layer.w[1:,:]
                    delta = np.array(delta, ndmin=2)

                    layer.w[0,:] = layer.w[0,:] + self.eta * np.mean(delta)
                    layer.w[1:,:] = layer.w[1:,:] + self.eta * layer_input.T @ delta

                

            loss = l2_cost[0](p_Y_training, out[-1][1])
            acc = self.get_accuracy(p_Y_validation, p_X_validation)
            print("acc: {}, loss: {}".format(acc, loss))

        return out[-1][1]

    def get_accuracy(self, p_Y_target, p_X):
        total = len(p_Y_target)
        predicted = self.predict(p_X)
        count = 0
        for c in range(total):
            if p_Y_target[c] == predicted[c]:
                count += 1

        return count / total * 100

    def predict(self, p_X):
        v_Y_input_layer_ = self.input_layer_.predict(p_X)
        v_X_hidden_layer_ = v_Y_input_layer_
        for v_hiddenlayer in self.hidden_layers_:
            v_Y_hidden_layer_ = v_hiddenlayer.predict(v_X_hidden_layer_)
            v_X_hidden_layer_ = v_Y_hidden_layer_
        v_X_output_layer_ = v_Y_hidden_layer_
        v_Y_output_layer_ = self.output_layer_.predict(v_X_output_layer_)
        return v_Y_output_layer_
