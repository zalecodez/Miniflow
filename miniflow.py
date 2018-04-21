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


    def forward(self):
        '''
        Forward Propogation
        '''

        raise NotImplemented
    

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


class Add(Node):
    def __init__(self, x, y):
        Node.__init__(self,[x,y])

    def forward(self):
        self.value = sum([n.value for n in self.inbound_nodes])

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

def forward_pass(output_node, sorted_nodes):
    '''
    Performs a forward pass through the list of sorted nodes
    '''
    for n in sorted_nodes:
        n.forward()
    return output_node.value
