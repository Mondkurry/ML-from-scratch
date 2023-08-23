import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from MLUtils import loadingAnimation, printPurple, plot_function, displayPower
import random
from graphviz import Digraph, render

# * Value class
class Value:
    """
    Value class that stores a single scalar value and its gradient, also added functions to be able to add/multiply objects.
    
    Attributes:
        value (float): The value of the object
        grad (float): The gradient of the object
        _prev (set): The set of all the previous objects
        _operation (str): The operation that was used to get the value
        _backward (function): The function that is used to back propagate the gradient
        
    Operations:
        __add__: Adds two values together
        __mul__: Multiplies two values together
        __truediv__: Divides two values together
        __pow__: Raises a value to a power
        __neg__: Negates a value

    Back Propagation:
        backProp: Back propagates the gradient through the graph
    
    Activation Functions:
        relu: Applies the ReLU activation function
        tanh: Applies the Tanh activation function
        sigmoid: Applies the Sigmoid activation function
        
    Repr: 
        Returns Value({label}: Data: {value}, Grad: {grad})
    """
    
    def __init__(self, value, _children=(), _op = "", **kwargs):
        self.label = kwargs["label"] if "label" in kwargs else ""
        self.value = value
        self.grad = 0.0
        self._prev = set(_children)
        self._operation = _op
        self._backward = lambda: None
        
    # * Arithmatic Operations
        
    def __add__(self, other) -> "Value":
        other = other if isinstance(other, Value) else Value(other) # checks if it is a value at first and if not, it creates a value
        out = Value(self.value + other.value, (self, other), "+")   # adds just the values of the Value data types
        
        def backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = backward
        
        return out
    
    def __mul__(self, other) -> "Value":
        other = other if isinstance(other, Value) else Value(other) # checks if it is a value at first and if not, it creates a value
        out = Value(self.value * other.value, (self, other), "*")
        
        def backward():
            self.grad += other.value * out.grad
            other.grad += self.value * out.grad
        out._backward = backward
        return out
    
    def __truediv__(self, other) -> "Value":
        return self.__mul__(other**-1)
    
    def __pow__(self, other) -> "Value":
        assert isinstance(other, (int, float)), "Power must be a number"
        out = Value(self.value**other, (self,), displayPower(self.label if self.label != "" else "(" + str(round(self.value, 6)) + ")", other.label if isinstance(other, Value) and other.label != "" else str(other)))
        
        def backward():
            self.grad += (other * self.value**(other-1)) * out.grad
        out._backward = backward
        
        return out
    
    def __round__(self, other) -> "Value":
        return Value(self.value.__round__(other))
    
    def __neg__(self):
        return self * -1
    
    # * Reverse Arithmatic Operations
        
    def __radd__(self, other) -> "Value":
        return self.__add__(other)
    
    def __rmul__(self, other) -> "Value":
        return self.__mul__(other)
    
    def __rtruediv__(self, other) -> "Value":
        return self.__truediv__(other)
    
    def __rpow__(self, other) -> "Value":
        return self.__pow__(other)
    
    # * Activation Functions
        
    def relu(self) -> "Value":
        out = Value(0 if self.value < 0 else self.value, (self,), "ReLU")
        
        def backward():
            self.grad += (out.value > 0) * out.grad
        out._backward = backward
        
        return out
    
    def tanh(self) -> "Value":
        out = Value(np.tanh(self.value), (self,), "Tanh")
        
        def backward():
            self.grad += (1 - out.value**2) * out.grad
        out._backward = backward
        
        return out
    
    def sigmoid(self) -> "Value":
        out = Value(1/(1+np.exp(-self.value)), (self,), "Sigmoid")
        
        def backward():
            self.grad += out.value * (1 - out.value) * out.grad
        out._backward = backward
        
        return out
        
    # * Back Propagation
    
    def backProp(self) -> None: 
        topo = []
        visited = set()
        def build_topo(v):  # Dont really understand the topo map but need to dive into it further
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __repr__(self):
        return f'Value({self.label + ": " if self.label != "" else self.label}Data: {self.value}, Grad: {self.grad})' # Didnt even know you could add functions like that but glad I tried it


class Visualizer():
    def __init__(self, root, rankdir = 'LR'):
        self.root = root
        self.nodes, self.edges = self.trace()
        self.format = 'pdf'
        self.rankdir = rankdir
    
    def trace(self):
            nodes, edges = set(), set()
            
            def build(current):
                if current not in nodes:
                    nodes.add(current)
                    for child in current._prev:
                        edges.add((child, current))
                        build(child)
            build(self.root)
            
            return nodes, edges
    
    def __call__(self, output = False):
        drawing = Digraph(format=self.format)
        drawing.attr(rankdir=self.rankdir)  # Set the rankdir attribute here
        
        for node in self.nodes:
            uid = str(id(node))
            label = "{%s | data %.4f | grad %.4f}" % (node.label, node.value, node.grad)
            drawing.node(name=uid, label=label, shape='record')
            
            if node._operation:
                drawing.node(name=uid + node._operation, label=node._operation)
                drawing.edge(uid + node._operation, uid)
                    
        for node1, node2 in self.edges:
            drawing.edge(str(id(node1)), str(id(node2)) + node2._operation)
            
        return drawing
    
class Neuron:
    '''
    A class to represent a neuron in a neural network.
    
    Attributes:
        weights (list): A list of weights for each input.
        bias (float): A bias value.
        activation (string): The activation function to use.
                Options: 'relu', 'sigmoid', 'tanh'

    '''
    
    def __init__(self, numInp, activation='relu'):
        self.weights = [Value(random.uniform(-1, 1), label = "wi") for i in range(numInp)]
        self.bias = Value(0, label='bias')
        self.activation = activation
        self.inputs = []

    def __call__(self, inputs):
        for i in inputs:
            i = Value(i)
            self.inputs.append(i)
            i.label = 'xi'
            
        wixiTotal = Value(0) ; wixiTotal.label = "wixi"
        for wi, xi in zip(self.weights, inputs):
            wixi = wi * xi ; wixi.label = "wixi"
            wixiTotal += wixi
            wixiTotal.label = "wixi"
        
        total = wixiTotal + self.bias
        total.label = 'total'
        
        if self.activation == 'relu':
            return total.relu()
        elif self.activation == 'sigmoid':
            return total.sigmoid()
        elif self.activation == 'tanh':
            return total.tanh()
        else:
            return total
        
    def parameters(self):
        return self.weights + [self.bias]
    
    def __repr__(self):
        return f'\nInputs: \t{self.inputs} \nWeights: \t{list(map(lambda x: round(x, 4), self.weights))} \nBias: \t\t{[self.bias]}\n'
    
class Layer:
    def __init__(self, numInput, numNeurons): # numNeurons is the same as specifying the number of outputs
        self.Neurons = [Neuron(numInput) for i in range(numNeurons)]
        
    def __call__(self, x):
        out = [Neuron(x) for Neuron in self.Neurons]
        return out[0] if len(out) == 1 else out # Figuring out Andrej's code here
    
    def parameters(self):
        return [p for n in self.Neurons for p in n.parameters()]
    
    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.Neurons)}]"
    
class MLP():
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"