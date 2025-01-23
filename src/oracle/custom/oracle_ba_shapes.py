from src.core.oracle_base import Oracle 

import numpy as np
import networkx as nx
from networkx.algorithms import isomorphism

class BAShapesOracle(Oracle):

    def init(self):
        super().init()
        self.model = ""

    def real_fit(self):
        pass

    def _real_predict(self, data_instance):
        return self.detect_house_in_graph(nx.from_numpy_array(data_instance.data))
        
    def _real_predict_proba(self, data_instance):
        # softmax-style probability predictions
        if self._real_predict(data_instance):
            return np.array([0,1])
        else:
            return np.array([1,0])
        
    def fwd(self, *args):
        pass


    def detect_house_in_graph(self, G):
        # Define the house subgraph structure (5 nodes, 6 edges)
        house_edges = [(0, 1), (1, 2), (2, 3), (3, 0),  # Square base
                    (0, 4), (2, 4)]                  # Roof connections

        house_graph = nx.Graph()
        house_graph.add_edges_from(house_edges)
        
        # Check for subgraph isomorphism anywhere in G
        GM = isomorphism.GraphMatcher(G, house_graph)
        
        for _ in GM.subgraph_isomorphisms_iter():
            return 1  # Found at least one instance
        
        return 0  # No house shape found
        
    