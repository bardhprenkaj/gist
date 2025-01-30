import pickle
import sys
import os

current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
src_dir = os.path.dirname(parent_dir)

sys.path.append(src_dir)

store_path = 'data/cache/datasets/BAShapes-58f34e6eebec1f08754e050bc0c64a6a'
with open(store_path, 'rb') as f:
    dump = pickle.load(f)
    '''self.instances = dump['instances']
    self.splits = dump['splits']
    #self.local_config = dump['config']
    self.node_features_map = dump['node_features_map']
    self.edge_features_map = dump['edge_features_map']
    self.graph_features_map = dump['graph_features_map']
    self._num_nodes = dump['num_nodes']
    self._class_indices = dump['class_indices'] '''

num_nodes = 0
num_edges = 0
for instance in dump['instances']:
    num_nodes += instance.data.shape[0]
    num_edges += len(instance.data.nonzero()[0]) // 2

print(f'avg nodes = {num_nodes/len(dump["instances"])}')
print(f'avg edges = {num_edges/len(dump["instances"])}')