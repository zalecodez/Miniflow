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
        raise NotImplemented


