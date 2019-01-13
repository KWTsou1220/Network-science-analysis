from nltk.corpus import stopwords

import networkx as nx
import numpy as np
import tqdm
import tensorflow as tf
import math

def read_train_data(path):
    '''
    Loading data
    Input:
        path: location of file in string
    Output:
        G: network.DiGraph
        node_paris: numpy array of each node pairs of shape [data_size, 2]
        t: target of the node pairs (0 or 1) in numpy array of shape [data_size, ]
    '''
    G = nx.DiGraph()
    nodes = []
    links = []
    with open(path, 'r') as f:
        count = 0
        for line in f:
            n_from, n_to, is_link = line.split(' ')
            n_from = int(n_from)
            n_to = int(n_to)
            is_link = int(is_link)
            nodes += [[n_from, n_to]]
            links += [is_link]
            if is_link==1:
                G.add_edge(n_from, n_to)
            else:
                G.add_node(n_from)
                G.add_node(n_to)
    return G, np.array(nodes), np.array(links)

def read_test_data(path):
    nodes = []
    with open(path, 'r') as f:
        count = 0
        for line in f:
            n_from, n_to = line.split(' ')
            n_from = int(n_from)
            n_to = int(n_to)
            nodes += [[n_from, n_to]]
    return np.array(nodes)

def extract_features(G, node_pairs, node_info):
    '''
    Extract features from the network.
    Input:
        G: networkx.Digraph()
        node_pairs: list of node pairs [(source, target), ...]
        node_info: the information of nodes {node: [public year, title, authors, name of journal, abstract]}
    Output:
        features: normalized features in numpy array 
    '''
    s_out_degree = G.out_degree(node_pairs[:, 0]) # source out degree
    s_in_degree = G.in_degree(node_pairs[:, 0]) # source in degree
    t_out_degree = G.out_degree(node_pairs[:, 1]) # target out degree
    t_in_degree = G.in_degree(node_pairs[:, 1]) # target in degree
    
    neighbors, successors, predecessors = get_neighbor(G)
    G_und = G.to_undirected()
    
    degree_centrality = nx.degree_centrality(G)
    in_degree_centrality = nx.in_degree_centrality(G)
    out_degree_centrality = nx.out_degree_centrality(G)
    eigen_centrality = nx.eigenvector_centrality(G)
    triangles = nx.triangles(G_und)
    clustering = nx.clustering(G_und)
    
    
    features = []
    for n_from, n_to in tqdm.tqdm(node_pairs):
        s_successor = successors[n_from] # set of successor of n_from
        s_predecessor = predecessors[n_from]
        s_neighbor = neighbors[n_from]
        t_successor = successors[n_to]
        t_predecessor = predecessors[n_to]
        t_neighbor = neighbors[n_to]
        
        features += [[
                # Topological Features
                s_out_degree[n_from], s_in_degree[n_from], t_out_degree[n_to], t_in_degree[n_to],
                s_out_degree[n_from]/(s_in_degree[n_from]+1e-6), s_in_degree[n_from]/(s_out_degree[n_from]+1e-6),
                t_out_degree[n_to]/(t_in_degree[n_to]+1e-6), t_in_degree[n_to]/(t_out_degree[n_to]+1e-6),
                t_in_degree[n_to]-s_in_degree[n_from],
                
                common_neighbor(s_successor, t_successor), common_neighbor(s_successor, t_predecessor),
                common_neighbor(s_predecessor, t_successor), common_neighbor(s_predecessor, t_predecessor),
                common_neighbor(s_neighbor, t_neighbor), 
                      
                jaccard_coeff(s_successor, t_successor), jaccard_coeff(s_successor, t_predecessor),
                jaccard_coeff(s_predecessor, t_successor), jaccard_coeff(s_predecessor, t_predecessor),
                jaccard_coeff(s_neighbor, t_neighbor),
                
                ada_coeff(s_successor, t_successor, neighbors), ada_coeff(s_successor, t_predecessor, neighbors),
                ada_coeff(s_predecessor, t_successor, neighbors), ada_coeff(s_predecessor, t_predecessor, neighbors),
                ada_coeff(s_neighbor, t_neighbor, neighbors),
                      
                preferential_attachment(s_successor, t_successor), preferential_attachment(s_successor, t_predecessor),
                preferential_attachment(s_predecessor, t_successor), preferential_attachment(s_predecessor, t_predecessor),
                preferential_attachment(s_neighbor, t_neighbor),
                
                resource_allocation(G_und, n_from, n_to),
                
                degree_centrality[n_from], degree_centrality[n_to],
                in_degree_centrality[n_from], in_degree_centrality[n_to],
                out_degree_centrality[n_from], out_degree_centrality[n_to],
                eigen_centrality[n_from], eigen_centrality[n_to], 
                triangles[n_from], triangles[n_to], 
                clustering[n_from], clustering[n_to], 
                
                # Attribute Features
                node_info[n_to][0]-node_info[n_from][0], # difference of publication year
                len(node_info[n_to][1].intersection(node_info[n_from][1])), # number of common words in title
                len(node_info[n_to][2].intersection(node_info[n_from][2])), # number of common authors
                int(node_info[n_to][3]==node_info[n_from][3]), # publich in the same journal or not
                len(node_info[n_to][4].intersection(node_info[n_from][4])), #number of common words in abstract
                     ]]
    return data_normalization(np.array(features))

def get_neighbor(G):
    '''
    Obtaining the successors, predecessors and the union of successors and predecessors.
    Input:
        G: networkx.DiGraph()
    Output:
        neighbors: dict{node: list of union of successors and predecessors}
        successors: dict{node: list of union of successors}
        predecessors: dict{node: list of union of predecessors}
    '''
    nodes = G.nodes()
    predecessors = {}
    successors = {}
    neighbors = {}
    for node in nodes:
        successors[node] = set(G.successors(node))
        predecessors[node] = set(G.predecessors(node))
        neighbors[node] = successors[node].union(predecessors[node])
    return neighbors, successors, predecessors

def common_neighbor(a, b): # set a and set b
    '''
    Obtain number of common neighbors
    Input:
        a: set of neighbors of n_from
        b: set of neighbors of n_to
    Output:
        number of neighbors in int
    '''
    return len(a.intersection(b))
def jaccard_coeff(a, b):
    '''
    Obtain jaccard coefficient given two set of neighbors
    Input:
        a: set of neighbors of n_from
        b: set of neighbors of n_to
    Output:
        jaccard coefficient in float
    '''
    return len(a.intersection(b))/(len(a.union(b)) + 1e-6 )
def ada_coeff(a, b, neighbors):
    '''
    Obtain adamic/adar index given two set of neighbors
    Input:
        a: set of neighbors of n_from
        b: set of neighbors of n_to
    Output:
        adamic/adar in float
    '''
    z = a.intersection(b)
    coeff = 0
    for _z in z:
        coeff += 1/np.log(len(neighbors[_z]))
    return coeff
def preferential_attachment(a, b):
    '''
    Obtain preferential attachment given two set of neighbors
    Input:
        a: set of neighbors of n_from
        b: set of neighbors of n_to
    Output:
         preferential attachment in int
    '''
    return len(a)*len(b)

def build_text_dict(path):
    '''
    Build a dictionary of node information
    Input:
        path: location of node_informacion file in string 
    Output:
        node_info: the information of nodes {node: [public year, title, authors, name of journal, abstract]}
    '''
    node_info = {}
    with open(path, encoding='utf8') as f:
        for line in f:
            info = line.split(',')
            _id = int(info[0])
            year = int(info[1])
            title = remove_stop_words(info[2], stopwords.words('english'))
            authors = info[3:-2]
            authors = set([author.replace('"', '').replace(' ', '') for author in authors])
            journal = info[-2]
            abstract = remove_stop_words(info[-1][0:-1], stopwords.words('english'))
            node_info[_id] = [year, title, authors, journal, abstract]
    return node_info

def undirected_shortest_path(G, s, t):
    '''
    Find the undirected shortest path
    Input:
        G: networkx.Digraph or networkx.Graph
        s: source node
        t: target node
    Output:
        path_length: distance of shortest path
    '''
    if G.is_directed():
        G_und = G.to_undirected()
    else:
        G_und = G
    path_length = 0
    try:
        path_length = nx.dijkstra_path_length(G_und, source=s, target=t)
    except:
        path_length =  len(G.nodes())
    return path_length

def directed_shortest_path(G, s, t):
    '''
    Find the directed shortest path
    Input:
        G: networkx.Digraph
        s: source node
        t: target node
    Output:
        path_length: distance of shortest path
    '''
    path_length = 0
    try:
        path_length = nx.dijkstra_path_length(G, source=s, target=t)
    except:
        path_length =  len(G.nodes())
    return path_length

def resource_allocation(G_und, s, t):
    '''
    Find the resource allocation index
    Input:
        G_und: networkx.Graph
        s: source node
        t: target node
    Output:
        resource allocation index
    '''
    list(nx.resource_allocation_index(G_und, [(s, t)]))[0][2]
    return nx.resource_allocation_index(G_und, [(s, t)]).__next__()[2]

def remove_stop_words(text, stop_words):
    '''
    Remove stop words in the text
    Input:
        text: string
        stop_words: list of stop words
    Output:
        filterd_text: list of string
    '''
    text = text.split(' ')
    filtered_text = [word for word in text if word not in stop_words]
    return set(filtered_text)

def data_normalization(features):
    '''
    Normalized data
    Input:
        features: numpy array [data_size, feature_size]
    Output:
        normalized features with same shape
    '''
    data_size = features.shape[0]
    mean = np.sum(features, axis=0)/data_size
    var = np.sum((features-mean)**2, axis=0)/data_size
    features_norm = np.divide(features-mean, np.sqrt(var)+1e-6)
    return features_norm