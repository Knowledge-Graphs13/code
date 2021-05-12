import numpy as np
import networkx as nx
from shapes import *

class SyntheticGraph():
  ###  shapes are put on top of the spanning tree of a graph
  ### This function creates a graph from a list of building blocks by addiing edges between blocks
  #### INPUT:
  #### -------------
  #### width_basis: width (in terms of number of nodes) of the basis
  #### basis_type: (torus, string, or cycle)
  #### list_shapes: list of shape list (1st arg: type of shape, next args: args for building the shape, except for the start)

  #### OUTPUT:
  #### -------------
  #### shape: a nx graph with the particular shape:
  def __init__(self, n_base, dict_shape, m
                 ):
    self.G = nx.barabasi_albert_graph(n_base, m, seed=None)
    self.community_colours = [0] * n_base
    self.labels = [0] * n_base
    self.shape_colour = [0] * n_base

    N = n_base
    for shape_nb, k in enumerate((dict_shape).keys()):
        for u in range(dict_shape[k][0]):
              
              args = [N, [self.community_colours[-1] + 1]]  + list(dict_shape[k][1:]) 
              S = eval(k)(*args)
              S.labels = [shape_nb + 1 + 100 * ll  for ll in S.labels]
              S.shape_colour = [shape_nb + 1] * S.n_nodes
              S.community_colours = [self.community_colours[-1] +1 + ll  for ll in S.community_colours]
              self.G.add_nodes_from(S.G.nodes())
              self.G.add_edges_from(S.G.edges())
              self.community_colours += S.community_colours
              self.shape_colour += S.shape_colour
              self.labels += S.labels
              self.G.add_edges_from([(np.random.choice(19),  N)])
              print([N, len(S.labels), len(self.labels)])
              N += S.n_nodes
              print(self.labels)
    self.n_nodes = N


  def draw(self, colour_by=None, cmap0=plt.cm.Set1):
        if colour_by == "label":
            nx.draw_networkx(self.G, node_color=self.labels, cmap=cmap0)
        elif colour_by == "community":
            nx.draw_networkx(self.G, node_color=self.community_colours, cmap=cmap0)
        elif colour_by == "shape":
            nx.draw_networkx(self.G, node_color=self.shape_colour, cmap=cmap0)
        else:
            nx.draw_networkx(self.G,cmap=cmap0)

  def saveplot(self, filename):
        plt.savefig(filename)