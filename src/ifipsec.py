# -*- coding: utf-8 -*-

"""
Code to run the IFIPSEC FbApp paper simulations

"""

import numpy as np
import networkx as nx
import random


def array_to_dict(arr):
    """
    Converts array to dictionary (positions as keys)

    :param arr: vector
    :return: dictionary
    """
    return {i: arr[i] for i in range(len(arr))}


def dict_to_array(dict):
    """
    Converts dictionary to array (keys as positions,
    keys must be consecutive integers from 0 to len-1)

    :param dict: dictionary
    :return: vector
    """
    return [dict[i] for i in range(len(dict.values()))]


def ba_mean_degrees(n, mean_degree):
    """
    Computes the best m to generate a ba graph
    with a given mean degree

    :param n: int Number of nodes of the graph
    :param mean_degree: int Expected mean degree
    :return: int or None
    """

    ms = range(1, 1000)
    prev_mean_deg = -1000
    for m in ms:
        g = nx.barabasi_albert_graph(n, m, seed=None)
        this_mean_deg = np.mean(nx.degree(g).values())
        if this_mean_deg > mean_degree:
            if this_mean_deg - mean_degree < mean_degree - prev_mean_deg:
                return m
            else:
                return m - 1
        prev_mean_deg = this_mean_deg

    return None

m = None
ws_p = 0.01  # adjust ws rewiring prob


def generate_graph(n, expected_degree, model="ba"):
    """
    Generates a graph with a given model and expected_mean
    degree

    :param n: int Number of nodes of the graph
    :param expected_degree: int Expected mean degree
    :param model: string Model (ba, er, or ws)
    :return: networkx graph
    """

    global m
    global ws_p

    g = None
    if model == "ba":
        # BA expected avg. degree? m = ba_mean_degrees()
        if m is None:
            m = ba_mean_degrees(n, expected_degree)
        g = nx.barabasi_albert_graph(n, m, seed=None)
    if model == "er":
        # ER expected avg. degree: d = p*(n-1)
        p = float(expected_degree) / float(n - 1)
        g = nx.erdos_renyi_graph(n, p, seed=None, directed=False)
    if model == "ws":
        # WS expected degree == k
        g = nx.watts_strogatz_graph(n, expected_degree, ws_p)

    return g


def assign_app_installed(g, q, model="uniform"):
    """
    Assigns app installations randomly following a given
    model (uniform, prop_friends_installed, yu_pu).
    Sets nodes' property "app installed"

    :param g: networkx graph
    :param q: float Fraction of users expected to install the app
    :param model: string App installation model
    :return: networkx graph, nodes have the property "app installed" set
    """

    if model == "uniform":
        app_installed = np.random.choice([True, False], g.number_of_nodes(), p=[q, 1 - q])
        nx.set_node_attributes(g, 'app_installed', array_to_dict(app_installed))

    if model == "prop_friends_installed":
        thisq = 0
        initial = [False for _ in range(g.number_of_nodes())]
        nx.set_node_attributes(g, 'app_installed', array_to_dict(initial))
        while thisq < q:
            num_myfriends_installed = compute_friends_installed(g, count=True, set_attribute=False)
            nodes_installed = nx.get_node_attributes(g, 'app_installed')
            targets = []
            for k, v in nodes_installed.items():
                if not v:
                    for _ in range(num_myfriends_installed[k] + 1):
                        targets.append(k)

            new_installation = random.choice(targets)
            nodes_installed[new_installation] = True
            nx.set_node_attributes(g, 'app_installed', nodes_installed)
            thisq = property_percentage(g, 'app_installed')

    return g


def compute_friends_installed(g, count=False, set_attribute=True):
    """
    Analyzes neighborhood app installation ("app_installed" attribute).
    Sets nodes' "at_least_one_friend" property (if set_attribute=True)

    :param g: networkx graph
    :param count: boolean,
        if True, computes (for each node) the number of neighbors that
            have the app installed;
        otherwise, indicates whether any of the friends have the app installed
    :param set_attribute: boolean,
        if True, sets nodes' "at_least_one_friend" property and returns the graph;
        otherwise, returns a vector with the results
    :return: networkx graph or vector
    """

    myfriends_installed = {}
    for n, d in g.nodes_iter(data=True):
        fs = g.neighbors(n)
        myfriends_installed[n] = False
        for f in fs:
            if g.node[f]['app_installed']:
                if count:
                    myfriends_installed[n] += 1
                else:
                    myfriends_installed[n] = True
                    break

    if set_attribute:
        nx.set_node_attributes(g, 'at_least_one_friend', myfriends_installed)
        return g
    else:
        return myfriends_installed


def property_percentage(g, property_name):
    """
    Computes the percentage of nodes having property_name
    property set to True

    :param g: networkx graph
    :param property_name: string Name of the property to evaluate
    :rtype: float
    """

    p = nx.get_node_attributes(g, property_name).values()
    return float(sum(p)) / float(len(p))


def evaluate_model(n, expected_degree, qs, num_graphs, graph_model="ba", adoption_model="uniform"):
    """
    Evaluates a given configuration:
        1. Generates num_graphs graphs of n nodes with the given graph_model and expected degree
        2. Assigns installations given adoption_model and desired fraction of users
        3. Computes how many users have a friend with the app installed

    :param n: int Number of nodes
    :param expected_degree: int Expected degree
    :param qs: float Fraction of users with the app installed
    :param num_graphs: int Number of graphs to generate
    :param graph_model: string Graph model (ba, er, or ws)
    :param adoption_model: string Adoption model (uniform, prop_friends_installed, yu_pu)
    :return: vector Fraction of users with the app installed and fraction of users with friends installation
    """

    avg_app_installed = []
    avg_friend_installed = []

    for q in qs:
        app_installed = []
        friend_installed = []

        for i in range(num_graphs):
            g = generate_graph(n, expected_degree, graph_model)
            g = assign_app_installed(g, q, adoption_model)
            g = compute_friends_installed(g)
            app_installed.append(property_percentage(g, 'app_installed'))
            friend_installed.append(property_percentage(g, 'at_least_one_friend'))

        avg_app_installed.append(np.mean(app_installed))
        avg_friend_installed.append(np.mean(friend_installed))

    return [avg_app_installed, avg_friend_installed]


if __name__ == "__main__":

    # Number of nodes
    N = 100
    # Expected mean degree of the graph
    Expected_degree = 30
    # Number of graphs to generate with each configuration
    Num_graphs = 5
    # Expected fraction of users with the app installed
    Qs = np.arange(0, 1.01, 0.05)

    # Evaluate models
    [avg_app_installed_er_unif, avg_friend_installed_er_unif] = evaluate_model(N, Expected_degree, Qs, Num_graphs,
                                                                               graph_model="er",
                                                                               adoption_model="uniform")
    [avg_app_installed_ba_unif, avg_friend_installed_ba_unif] = evaluate_model(N, Expected_degree, Qs, Num_graphs,
                                                                               graph_model="ba",
                                                                               adoption_model="uniform")
    [avg_app_installed_ws_unif, avg_friend_installed_ws_unif] = evaluate_model(N, Expected_degree, Qs, Num_graphs,
                                                                               graph_model="ws",
                                                                               adoption_model="uniform")

    [avg_app_installed_er_prop, avg_friend_installed_er_prop] = evaluate_model(N, Expected_degree, Qs, Num_graphs,
                                                                               graph_model="er",
                                                                               adoption_model="prop_friends_installed")
    [avg_app_installed_ba_prop, avg_friend_installed_ba_prop] = evaluate_model(N, Expected_degree, Qs, Num_graphs,
                                                                               graph_model="ba",
                                                                               adoption_model="prop_friends_installed")
    [avg_app_installed_ws_prop, avg_friend_installed_ws_prop] = evaluate_model(N, Expected_degree, Qs, Num_graphs,
                                                                               graph_model="ws",
                                                                               adoption_model="prop_friends_installed")
