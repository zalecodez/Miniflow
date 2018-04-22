import numpy as np
class Node (object):
    def __init__(self, inbound_nodes=[]):

        #inputs to this node
        self.inbound_nodes = inbound_nodes

        #outputs of this node
        self.outbound_nodes  = []

        #for each input node, add this node as an outbound node
        for n in self.inbound_nodes:
            n.outbound_nodes.append(self)

        self.value = None
        self.gradients = {}


    def forward(self):
        '''
        Forward Propogation
        '''
        raise NotImplementedError
    
    def backward(self):
        '''
        Backpropogation
        '''
        raise NotImplementedError

    

class Input(Node):
    '''
    Subclass of Node. 
    Input has no inbound nodes
    '''

    def __init__(self):
        Node.__init__(self) #no inbound nodes

    def forward(self, value=None):
        if value is not None:
            self.value = value
    
    def backward(self):
        self.gradients = {self:0}
        for n in self.outbound_nodes:
            grad_cost = n.gradients[self]
            self.gradients[self] += grad_cost*1

class Add(Node):
    def __init__(self, *inputs):
        Node.__init__(self, inputs)

    def forward(self):
        self.value = sum([n.value for n in self.inbound_nodes])

class Mul(Node):
    def __init__(self, *inputs):
        Node.__init__(self, inputs)

    def forward(self):
        self.value = 1
        for n in self.inbound_nodes:
            self.value *= n

class Linear(Node):
    def __init__(self, inputs, weights, bias):
        Node.__init__(self, [inputs, weights, bias])

    def forward(self):
        inputs = self.inbound_nodes[0].value
        weights = self.inbound_nodes[1].value
        bias = self.inbound_nodes[2].value

        self.value = np.dot(inputs, weights) + bias
   
    def backward(self):
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}
        for n in self.outbound_nodes:
            grad_cost = n.gradients[self]
            #gradient wrt inputs
            self.gradients[self.inbound_nodes[0]] += np.dot(grad_cost, self.inbound_nodes[1].value.T)

            #gradient wrt W
            self.gradients[self.inbound_nodes[1]] += np.dot(self.inbound_nodes[0].value.T, grad_cost)

            #gradient wrt b
            self.gradients[self.inbound_nodes[2]] += np.sum(grad_cost, axis=0, keepdims=False) 

class Sigmoid(Node):
    def __init__(self, node):
        Node.__init__(self, [node])
    
        self.value = 8
    def _sigmoid(self, x):
        x = 1./(1+np.exp(-x))
        return x

    def forward(self):
        x = self.inbound_nodes[0].value
        self.value = self._sigmoid(x)
    def backward(self):
        """
        Calculates the gradient using the derivative of
        the sigmoid function.
        """
        # Initialize the gradients to 0.
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}

        # Cycle through the outputs. The gradient will change depending
        # on each output, so the gradients are summed over all outputs.
        for n in self.outbound_nodes:
            # Get the partial of the cost with respect to this node.
            grad_cost = n.gradients[self]
            """
            TODO: Your code goes here!

            Set the gradients property to the gradients with respect to each input.

            NOTE: See the Linear node and MSE node for examples.
            """

            thisgrad = self.value*(1-self.value)
            self.gradients[self.inbound_nodes[0]] += grad_cost*thisgrad
        
class MSE(Node):
    def __init__(self, y, a):
        Node.__init__(self, [y,a])

    def forward(self):
        self.y = self.inbound_nodes[0].value.reshape(-1,1)
        self.a = self.inbound_nodes[1].value.reshape(-1,1)

        self.value = np.mean(np.square(self.y - self.a))


    def backward(self):
        self.m = self.y.shape[0]
        self.gradients[self.inbound_nodes[0]] = 2*(self.y-self.a)/self.m
        self.gradients[self.inbound_nodes[1]] = -2*(self.y-self.a)/self.m

def forward_pass(output_node, sorted_nodes):
    '''
    Performs a forward pass through the list of sorted nodes
    '''
    for n in sorted_nodes:
        n.forward()
    return output_node.value

def forward_and_backward(graph):
    for n in graph:
        n.forward()

    for n in graph[::-1]:
        n.backward()

def sgd_update(trainables, learning_rate=1e-2):
    for t in trainables:
        t.value = t.value - learning_rate*t.gradients[t]
        

def topological_sort(feed_dict):
    '''
    Sort generic nodes in topological order using Kahn's Algorithm
    feed_dict = dictionary in which the keys an Input node and the value is the value for that node
    '''

    input_nodes = [n for n in feed_dict.keys()]

    #create graph G
    G = {}
    nodes = [n for n in input_nodes] 
    while len(nodes) > 0:
        n = nodes.pop(0)
        if n not in G:
            G[n] = {"in":set(), "out":set()}
        for m in n.outbound_nodes:
            nodes.append(m)
            if m not in G:
                G[m] = {"in":set(), "out":set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)

    S = set(input_nodes) #nodes with no incoming edge
    L = [] # sorted list of nodes
    while len(S) > 0:
        n = S.pop()

        #give inputs their values
        if isinstance(n, Input):
            n.value = feed_dict[n]

        #add node to the list since it has no remaining inbound nodes
        L.append(n)

        #look at all children of n... for each, if n is it's only parent, remove the link and add it to S
        while len(G[n]['out']) > 0:
            m = G[n]['out'].pop()
            G[m]['in'].remove(n)
            if len(G[m]['in']) == 0:
                S.add(m) 

    return L


