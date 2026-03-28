import numpy as np
import time
from collections import defaultdict

import networkx as nx

def load_dataset(filename):

    graph_dic = nx.DiGraph()

    with open(filename, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue

            parts = line.split()
            from_where = int(parts[0])
            to_where = int(parts[1])

            # add to dic
            graph_dic.add_edge(from_where, to_where)

    all_nodes = list(graph_dic.nodes())

    return graph_dic, all_nodes

def initialize_pagerank(nodes):

    N = len(nodes)

    ranks = {}

    for node in nodes:
        ranks[node] = 1 / N

    return ranks

def pagerank(graph_dic, all_nodes, damping=0.85, iterations=20):

    num_nodes = len(all_nodes)

    rankings = initialize_pagerank(all_nodes)

    dangling_nodes = []
    for n in all_nodes:
        if graph_dic.out_degree(n) == 0:
            dangling_nodes.append(n)

    for i in range(iterations):

        updated_ranking = {}

        dangling_sum = 0
        for n in dangling_nodes:
            dangling_sum += (rankings[n])

        for node in all_nodes:
            updated_ranking[node] = (1 - damping) / num_nodes
            updated_ranking[node] += damping * dangling_sum / num_nodes

        for node in graph_dic.nodes():

            out_links =  list(graph_dic.successors(node))

            if len(out_links) == 0:
                continue

            out_degree = len(out_links)
            share = rankings[node] / out_degree

            for to_where in out_links:
                updated_ranking[to_where] += damping * share

        rankings = updated_ranking

    return rankings

def pagerank_closed_form(graph_dic, all_nodes, damping=0.85):

    num_nodes = len(all_nodes)

    index = {}
    for i, node in enumerate(all_nodes):
        index[node] = i

    M = np.zeros((num_nodes, num_nodes))

    for from_where in graph_dic.nodes():

        out_links = list(graph_dic.successors(from_where))

        if len(out_links) == 0:
            continue

        for to_where in out_links:
            M[index[to_where], index[from_where]] = 1 / len(out_links)

    A = damping * M + (1 - damping) / num_nodes * np.ones((num_nodes, num_nodes))

    eigenvalues, eigenvectors = np.linalg.eig(A)

    eigen_index = np.argmax(eigenvalues)

    page_rank = np.abs(eigenvectors[:, eigen_index].real)

    page_rank = page_rank / np.sum(page_rank)

    return page_rank

if __name__ == "__main__":

    data = load_dataset("web-Google_10k.txt")

    graph_dict = data[0]
    nodes = data[1]

    num_nodes = len(nodes)
    num_edges = graph_dict.number_of_edges()

    iterative_ranks = pagerank(graph_dict, nodes, damping=0.85, iterations=20)

    closed_form_page_ranks = pagerank_closed_form(graph_dict, nodes)

    iter_vec = []
    for n in nodes:
        iter_vec.append(iterative_ranks[n])
    iter_vec = np.array(iter_vec)
    error = np.linalg.norm(iter_vec - closed_form_page_ranks)

    sorted_iter = sorted(iterative_ranks.items(), key=lambda x: x[1], reverse=True)



    print("Iterative PageRank")
    for i, (node, score) in enumerate(sorted_iter[:20], start=1):
        print(f"{i:2d}. Node {node:8d}  Score: {score:.8f}")



    print("\nClosed form page rank")
    closed_pairs = list(zip(nodes, closed_form_page_ranks))
    closed_pairs.sort(key=lambda x: x[1], reverse=True)

    for i, (node, score) in enumerate(closed_pairs[:20], start=1):
        print(f"{i:2d}. Node {node:8d}  Score: {score:.8f}")

    print("L2 Norm Difference:", error)
