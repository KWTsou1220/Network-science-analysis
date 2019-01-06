import networkx as nx
import numpy as np
import tqdm

def read_train_data(path):
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

def extract_features(G, node_pairs):
    s_out_degree = G.out_degree(node_pairs[:, 0]) # naming: source out degree
    s_in_degree = G.in_degree(node_pairs[:, 0])
    t_out_degree = G.out_degree(node_pairs[:, 1])
    t_in_degree = G.in_degree(node_pairs[:, 1])
    
    neighbors, successors, predecessors = get_neighbor(G)
    
    features = []
    for n_from, n_to in tqdm.tqdm(node_pairs):
        s_successor = successors[n_from]
        s_predecessor = predecessors[n_from]
        s_neighbor = neighbors[n_from]
        t_successor = successors[n_to]
        t_predecessor = predecessors[n_to]
        t_neighbor = neighbors[n_to]
        
        features += [[s_out_degree[n_from], s_in_degree[n_from], t_out_degree[n_to], t_in_degree[n_to],
                      s_out_degree[n_from]/(s_in_degree[n_from]+1e-6), s_in_degree[n_from]/(s_out_degree[n_from]+1e-6),
                      t_out_degree[n_to]/(t_in_degree[n_to]+1e-6), t_in_degree[n_to]/(t_out_degree[n_to]+1e-6),
                      
                      common_neighbor(s_successor, t_successor), common_neighbor(s_successor, t_predecessor),
                      common_neighbor(s_predecessor, t_successor), common_neighbor(s_predecessor, t_predecessor),
                      common_neighbor(s_neighbor, t_neighbor), 
                      
                      jaccard_coeff(s_successor, t_successor), jaccard_coeff(s_successor, t_predecessor),
                      jaccard_coeff(s_predecessor, t_successor), jaccard_coeff(s_predecessor, t_predecessor),
                      jaccard_coeff(s_neighbor, t_neighbor), 
                      
                      #ada_coeff(s_successor, t_successor, successors), ada_coeff(s_successor, t_predecessor, successors),
                      #ada_coeff(s_predecessor, t_successor, successors), ada_coeff(s_predecessor, t_predecessor, successors),
                      #ada_coeff(s_neighbor, t_neighbor, successors), 
                      
                      #ada_coeff(s_successor, t_successor, predecessors), ada_coeff(s_successor, t_predecessor, predecessors),
                      #ada_coeff(s_predecessor, t_successor, predecessors), ada_coeff(s_predecessor, t_predecessor, predecessors),
                      #ada_coeff(s_neighbor, t_neighbor, predecessors),
                      
                      ada_coeff(s_successor, t_successor, neighbors), ada_coeff(s_successor, t_predecessor, neighbors),
                      ada_coeff(s_predecessor, t_successor, neighbors), ada_coeff(s_predecessor, t_predecessor, neighbors),
                      ada_coeff(s_neighbor, t_neighbor, neighbors),
                      
                      preferential_attachment(s_successor, t_successor), preferential_attachment(s_successor, t_predecessor),
                      preferential_attachment(s_predecessor, t_successor), preferential_attachment(s_predecessor, t_predecessor),
                      preferential_attachment(s_neighbor, t_neighbor),
                     ]]
    return np.array(features)

def get_neighbor(G):
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
    return len(a.intersection(b))
def jaccard_coeff(a, b):
    return len(a.intersection(b))/(len(a.union(b)) + 1e-6 )
def ada_coeff(a, b, neighbors):
    z = a.intersection(b)
    coeff = 0
    for _z in z:
        coeff += 1/np.log(len(neighbors[_z]))
    return coeff
def preferential_attachment(a, b):
    return len(a)*len(b)
