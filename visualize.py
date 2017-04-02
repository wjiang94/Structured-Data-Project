import numpy
import random
import networkx as nx
import matplotlib.pyplot as plt

def visualize(ker, threshold=0.3):
    
    '''
    L is the output kernel
    labels come form caltech_101_classes.txt
    singletons are omitted from the graph
    '''
    G = nx.Graph()
    weight = {}
    
    #initialize the weights            
    for i in range(len(ker)):
        a = list(ker[i])
        del a[i]
        if np.max(a) > threshold:
            weight[i] = {}
            for j in range(i+1,len(ker)):
                if ker[i][j] > threshold:
                    weight[i][j] = ker[i][j]
    print(weight.keys())
    
    #add nodes
    G.add_nodes_from(list(weight.keys()))
    
    #add edges if similarity > 0.5
    for i in weight.keys():
        for j in weight[i].keys():
            G.add_edges_from([(i,j,{'weight':ker[i][j]})])
    pos = nx.spring_layout(G)

    #load labels
    file = open('labels/caltech_101_classes.txt','r')
    lines = file.readline().strip().split(',')
    lines = [lines[i] for i in range(len(lines)) if i in weight.keys()]
    label = dict(zip(weight.keys(),lines))
    
    #drwa the graph
    plt.figure(figsize=(20,20))
    nx.draw_networkx_nodes(G,pos,node_color='grey',
                          node_size=500,alpha=0.8)

    nx.draw_networkx_edges(G,pos,width=1.0,alpha=0.5)
    nx.draw_networkx_labels(G,pos,label,font_size=16)
    plt.axis('off')
    plt.savefig('full.png')

    #show the figure
    #plt.show()
    
if __name__ == '__main__':
    
    """
    L here is randomly intialized
    """
    L = numpy.eye(30)
    for i in range(len(L)):
        for j in range(i+1,len(L)):
            L[i][j] = random.random()
  

    visualize(L)

