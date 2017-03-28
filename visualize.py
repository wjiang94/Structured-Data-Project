import numpy
import random
import networkx as nx
import matplotlib.pyplot as plt

def visualize(ker, label, threshold=0.6):
    
    """
    L is the output kernel
    labels come form caltech_101_classes.txt
    """
    G = nx.Graph()
    
    #add nodes
    G.add_nodes_from(list(range(len(ker))))

    #add edges if similarity > 0.5
    for i in range(len(ker)):
        for j in range(i+1,len(ker)):
            if ker[i][j] > threshold:
                G.add_edges_from([(i,j,{'weight':ker[i][j]})])
    pos = nx.spring_layout(G)

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
  
    #load labels
    file = open('labels/caltech_101_classes.txt','r')
    lines = file.readline().strip().split(',')
    labels = dict(zip(list(range(len(L))),lines))
  
    visualize(L, labels)
