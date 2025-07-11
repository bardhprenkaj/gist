from collections import OrderedDict
import copy
import json
import os
import torch
import numpy as np

def update_saved_pyg(input_file,output_file):
    old_model =  torch.load(input_file, map_location=torch.device('cpu'))
    fixed_model = OrderedDict([(k.replace("grpah", "graph"), v) if 'grpah' in k else (k, v) for k, v in old_model.items()])
    torch.save(fixed_model,output_file)

def sanitize_dir_pyg(based_dir,prefix,model_name='explainer'):
    for file in os.listdir(based_dir):
        if file.startswith(prefix):            
            model_file_name = os.path.join(based_dir,file,model_name)
            if os.path.exists(model_file_name):
                old_file_name = os.path.join(based_dir,file,"OLD_"+model_name)

                print("Sanitizing: "+model_file_name)

                os.rename(model_file_name, old_file_name)
                print("Renamed to: "+old_file_name)

                update_saved_pyg(old_file_name,model_file_name)
                print("Complete")

def unfold_confs(based_dir,out_dir,prefix,num_folds=10):
    for dir in os.listdir(based_dir):
        if dir.startswith(prefix) and os.path.isdir(os.path.join(based_dir,dir)):
            # os.makedirs(os.path.join(out_dir,dir), exist_ok=True)
            for sub_dir in os.listdir(os.path.join(based_dir,dir)):
                if os.path.isdir(os.path.join(based_dir,dir,sub_dir)):
                    os.makedirs(os.path.join(out_dir,dir,sub_dir), exist_ok=True)
                    print("Processing subfolder: "+os.path.join(based_dir,dir,sub_dir))
                    for conf_file in os.listdir(os.path.join(based_dir,dir,sub_dir)):
                        #print(conf_file)
                        in_file = os.path.join(based_dir,dir,sub_dir,conf_file)
                        out_file = os.path.join(out_dir,dir,sub_dir,conf_file)

                        with open(in_file, 'r') as config_reader:
                            configuration = json.load(config_reader)                                                    
                            for fold_id in range(num_folds):
                                current_conf =  copy.deepcopy(configuration)
                                for exp  in current_conf['explainers']:
                                    exp['parameters']['fold_id']=fold_id
                                
                                out_file = os.path.join(out_dir,dir,sub_dir,conf_file[:-5]+'_'+str(fold_id)+'.json')
                                with open(out_file, 'w') as o_file:
                                    json.dump(current_conf, o_file)
                                print(out_file)
                                
def pad_adj_matrix(adj_matrix, target_dimension):
    # Get the current dimensions of the adjacency matrix
    current_rows, current_cols = adj_matrix.shape
    # Calculate the amount of padding needed for rows and columns
    pad_rows = max(0, target_dimension - current_rows)
    pad_cols = max(0, target_dimension - current_cols)
    # Pad the adjacency matrix with zeros
    return np.pad(adj_matrix, ((0, pad_rows), (0, pad_cols)), mode='constant')

def pad_features(features, target_dimension):
    nodes, feature_dim = features.shape
    if nodes < target_dimension:
        rows_to_add = max(0, target_dimension - nodes)
        to_pad = np.zeros((rows_to_add, feature_dim))
        features = np.vstack([features, to_pad])
    return features
