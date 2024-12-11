import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from src.dataset.instances.graph import GraphInstance
from src.core.explainer_base import Explainer
from src.utils.cfg_utils import retake_oracle


class CFGNNCExplainer(Explainer):
    '''
    Explainer based on
    Lucic et al. CF-GNNExplainer Counterfactual Explanations for Graph Neural Networks
    https://arxiv.org/abs/2102.03322
    Algorithm: CF-GNNExplainer: given a node v = (Av, x) where f (v) = y, generate the minimal perturbation,  ̄v = (Āv, x), such that f (̄v) ≠ y. [5.4]
    '''
    
    def init(self):
        self.oracle = retake_oracle(self.local_config)
        
        local_params = self.local_config['parameters']
        self.α = local_params['alpha'] # α: Learning Rate
        self.K = local_params['K'] # K: Number of Iterations (to update pertubation matrices (P_hat, etc.))
        self.β = local_params['beta'] # β: Trade-off between Lpred and Ldist Eq.1 [4]
        self.γ = local_params['gamma'] # γ: Add missing edges to the adjacency matrix (γ ∈ [0, 1])

        # self.loss_fn = torch.nn.BCELoss()
        self.loss_fn = torch.nn.NLLLoss()
        
        assert ((isinstance(self.K, float) or isinstance(self.K, int)) and self.K >= 1)
        assert ((isinstance(self.α, float) or isinstance(self.α, int)) and self.α > 0)
        assert ((isinstance(self.β, float) or isinstance(self.β, int)) and self.β >= 0)
        assert ((isinstance(self.γ, float) or isinstance(self.γ, int)) and 0 <= self.γ <= 1)

        # print(self.α, self.K, self.β)


    def explain(self, instance):
        # self.A_v = torch.tensor(instance.data, dtype=torch.float64) # instance.data is the adjacency matrix
        self.A_v = torch.ones(instance.data.shape, dtype=torch.float64) # Assume full adjacency matrix and use Ldist to find best CF
        missing_edges = np.where(instance.data == 0) # Missing A_v edges

        self.node_features = torch.ones(instance.node_features.shape, dtype=torch.float64)
        missing_nodefeatures = np.where(instance.node_features == 0) # Missing node_features edges
        # print(missing_edges)
        # print(instance.node_features)

        # self.A_v = instance.data # instance.data is the adjacency matrix
        # print(self.A_v) # Debugging

        self.x = torch.tensor(instance.node_features, dtype=torch.float64) # feature vector for v
        # self.x = instance.node_features # feature vector for v

        self.v = (self.A_v, self.x)
        
        self.f_v = torch.tensor(self.oracle.predict(instance), dtype=torch.long) # Get GCN prediction
        # print(f"{Color.YELLOW}Initial prediction (f_v): {Color.RESET}{self.f_v}")  # Debugging

        '''f_v_GI = GraphInstance(
            id = 0,
            label = "Original Instance",
            data = instance.data,
            node_features = instance.node_features,
            # edge_features = edge_features_np,
            # edge_weights= edge_weights_np,
            edge_features = instance.edge_features,
            edge_weights = instance.edge_weights,
            graph_features= instance.graph_features
            )
        
        f_v_GI_pred = torch.tensor(self.oracle.predict(f_v_GI), dtype=torch.float64) # Get GCN prediction
        print(f"Initial prediction (f_v_GI_pred): {f_v_GI_pred}")  # Debugging       '''
        
        self.P_hat = torch.ones_like(self.A_v, requires_grad=False) # Initialization of P_hat
        self.P_hat[missing_edges] = self.γ # Adjacency matrix is full of ones and P stores the zero edges (i.e. inverting the roles of A_v and P)
        self.P_hat.requires_grad_(True)
        # print(self.P_hat, instance.data)
        # print(torch.equal(self.P_hat, torch.tensor(instance.data))) # Debugging: with γ != 0 → False

        self.P_node_hat = torch.tensor(instance.node_features, requires_grad=False) # Initialization of P_node_hat: Perturbation matrix for Node Features
        # self.P_node_hat[missing_nodefeatures] = self.γ
        self.P_node_hat.requires_grad_(True)
        # print(self.P_node_hat, self.node_features)

        self.v_bar_opt = (self.A_v, self.x) # Initializing optimal CF with the instance itself
        self.opt_flag = False
        edge_indices = np.where(instance.data != 0) # int array
        self.edge_weights_opt = instance.data[edge_indices] # real array

        for _ in range(int(self.K)):
            # print(f"Iteration: {_}")
            
            v_bar, valid_CF = self.__get_CF_example(instance)

            #print("Calculating loss")
            loss = self.__calculate_loss(v_bar, instance, valid_CF)
            # print("Ldist:", L_dist, self.β, L_pred)
            # print(self.N_v_bar_copy.grad, self.A_v_bar_copy.grad)
            # print(f"loss: {total_loss}")
            # print(self.N_v_bar.requires_grad, self.A_v_bar.requires_grad)
            # print(torch.autograd.grad(loss, self.N_v_bar_copy, retain_graph=True))
            # print(torch.autograd.grad(loss, self.A_v_bar_copy, retain_graph=True))
            # print(self.A_v.dtype, self.A_v_bar.dtype)
            # print(self.node_features.dtype, self.N_v_bar.dtype)
            # print(loss.grad_fn)  # Should display the final operation of the graph

            # Backpropagate the loss and compute gradient of P_hat
            loss.backward(retain_graph=True) # Retaining graph to sum future backward gradients
            
            # Manually retrieve and store gradients
            self.N_v_bar.grad = torch.autograd.grad(loss, self.N_v_bar, retain_graph=True)[0]; # print("N_v_bar.grad:", self.N_v_bar_grad)
            self.A_v_bar.grad = torch.autograd.grad(loss, self.A_v_bar, retain_graph=True)[0]; # print("A_v_bar.grad:", self.A_v_bar_grad)

            # print(self.g_v_logits.grad)
            # print(f"self.A_v_bar.shape: {self.A_v_bar.shape}")
            # print(self.node_features.grad.equal(self.x.grad))
            # print(self.node_features.shape)
            # print(f"self.a.grad: {self.a.grad}")
            # print(f"self.edge_index.grad: {self.edge_index.grad}")            
            # print(self.adj.grad)

            # print(self.edge_weights.grad.shape)
            # print(f"self.w grad: {self.w.grad}")
            # print(self.w.grad.shape)
            # print(f"Are self.w.grad equal to self.edge_weigths.grad: {self.w.grad.equal(self.edge_weights.grad)}")

            # print(self.x.grad)
            # self.A_v_bar.backward(torch.tensor([1]))

            '''new_grad = torch.zeros_like(self.w.grad)
            mask = self.w != 0
            new_grad[mask] = self.w.grad[mask]
            # If gradients were with zeroes but they are non-sparse
            '''
            # print(f"self.edge_indices.shape: {self.edge_indices[0].shape}")
            # print(f"self.node_indices.shape: {self.node_indices[0].shape}")

            '''print(f"self.A_v_bar: {self.A_v_bar}")
            print(f"self.A_v_bar.requires_grad: {self.A_v_bar.requires_grad}")
            print(f"self.w: {self.w}")
            print(f"self.w.requires_grad: {self.w.requires_grad}")
            print(f"self.w.grad: {self.w.grad}")  # Check if this is None at this point
            print(f"self.w_grads: {self.w_grads}")
            print(f"self.g_v_probabilities: {self.g_v_probabilities}")'''

            '''if valid_CF:
                torch.set_printoptions(threshold=10_000)
                print(self.P_hat.grad)#  != None)
                exit()'''
            if not valid_CF: # L_pred contributing to Total Loss
                # self.A_v_bar.grad = torch.rand(self.A_v_bar.shape, dtype=torch.double) * 1e-1 # (Incorrect!) Initialize A_v_bar gradients randomically
                self.A_v_bar.grad += torch.zeros_like(self.A_v_bar, dtype=torch.double) # Initialize A_v_bar gradients with zeroes
                # torch.set_printoptions(threshold=10_000)
                # print(self.A_v_bar.grad)
                # print(f"self.A_v_bar.grad.shape: {self.A_v_bar.grad.shape}")
                self.A_v_bar.grad[self.edge_indices] += self.w.grad # Adding gradients from self.w.grad to the positions specified in self.edge_indices, i.e. they are the non-zero and contributing edges to the classification
                # Adding L_pred contribution to the L_dist
                
                self.N_v_bar.grad += torch.zeros_like(self.N_v_bar, dtype=torch.double)
                # print(f"self.N_v_bar.grad.shape: {self.N_v_bar.grad.shape}")
                # print(f"self.x.grad.shape: {self.x.grad.shape}")             
                self.N_v_bar.grad += self.x.grad.clone() # Adding L_pred contribution to the L_dist

                # torch.set_printoptions(threshold=10_000)
                # print("self.A_v_bar.grad", self.A_v_bar.grad)
                
            self.A_v_bar.backward(torch.ones_like(self.A_v_bar)) # Perform backward pass
            self.N_v_bar.backward(torch.ones_like(self.N_v_bar))

            with torch.no_grad():  # Update without tracking the gradients further
                # print(f"Gradient of P_hat: {self.P_hat.grad}")  # Debugging to check gradients
                self.P_hat -= self.α * self.P_hat.grad  # Gradient step with learning rate
                self.P_node_hat -= self.α * self.P_node_hat
                # print(f"self.P_hat.grad: {self.P_hat.grad}")
                    
            self.P_hat.grad.zero_()
            self.P_node_hat.grad.zero_()

            '''print(k)
            if k == 3: exit()
            print(f"self.P_hat: {self.P_hat}")'''

            # print(f"iteraton {k} finished") # Debugging

            if valid_CF: break # Breaking at first CF found (Different from paper algorithm → Efficiency reason: avoiding extra loop iterations)
            
        
        # print("Exiting")
        # print(f"opt CF: {self.v_bar_opt}")
        
        # num_edges = torch.count_nonzero(self.v_bar_opt[0])
        # print(len(self.edge_weights_opt), num_edges)
        edge_indices = torch.where(self.v_bar_opt[0] != 0) # int tensor
        edge_weights = self.v_bar_opt[0][edge_indices] # real tensor
        edge_weights = edge_weights.clone().detach().numpy()
        # print(f"edge_weights.shape: {edge_weights.size}")
        # print(self.edge_weights_opt.dtype)

        # Passing the CF node features
        node_features = self.N_v_bar.clone().detach().numpy()
        
        try:
            edge_features = instance.edge_features[edge_indices]

            # print(f"instance.node_features.shape: {instance.node_features.shape}")
            # print(f"edge_weights.shape: {edge_weights.shape}")

            # print(edge_weights)
            # print(self.edge_weights_opt)

            v_bar_opt_GI = GraphInstance(
                id = instance.id,
                label = "Optimal CF",
                data = self.v_bar_opt[0].clone().detach().numpy(),
                node_features = node_features,
                edge_features = edge_features,
                edge_weights = edge_weights,
                # edge_weights= self.edge_weights_opt
                graph_features= instance.graph_features
                )
        except:
            v_bar_opt_GI = GraphInstance(
                id = instance.id,
                label = "Optimal CF",
                data = self.v_bar_opt[0].clone().detach().numpy(),
                node_features = node_features,
                # edge_features = edge_features,
                edge_weights = edge_weights,
                # edge_weights= self.edge_weights_opt
                graph_features= instance.graph_features
                )

        
        # print(f"{Color.MAGENTA}Opt predicted class: {Color.RESET}{self.oracle.predict(v_bar_opt_GI)}")
        return v_bar_opt_GI


    def __get_CF_example(self, instance):
        # P = (torch.sigmoid(self.P_hat) >= .5).type(torch.float) # Threshold on sigmoid of P_hat

        P_sigmoid = torch.sigmoid(self.P_hat) # Threshold on sigmoid of P_hat
        mask = (P_sigmoid >= .5).float().clone() # Hard mask
        P = P_sigmoid + (mask - P_sigmoid).detach() # Gradients can flow through P

        self.A_v_bar = P * self.A_v
        self.A_v_bar.fill_diagonal_(1) # Add self-loop Eq(4) [5.2]
        # self.A_v_bar_copy = torch.tensor(self.A_v_bar.data, requires_grad=True)

        v_bar_cand = (self.A_v_bar, self.x)
        # print("v_bar_cand", v_bar_cand) # Debugging

        # Count number of edges in A_v_bar
        # edges = torch.nonzero(self.A_v_bar)  # Get the indices of non-zero entries in the modified adjacency matrix    

        '''edge_weights = torch.zeros_like(self.A_v)
        edge_weights[edges[0], edges[1]] = instance.edge_weights[edges[0], edges[1]] # Taking only the weights of edges present after pertubation
        edge_weights = edge_weights.flatten()

        edge_features = torch.zeros_like(self.A_v)
        edge_features[edges[0], edges[1]] = instance.edge_features[edges[0], edges[1]] # Taking only the features of edges present after pertubation
        edge_features = edge_features.flatten()'''

        A_v_bar_np = self.A_v_bar.clone().detach().numpy(); # print(type(A_v_bar_np))  # Should be <class 'numpy.ndarray'>
        ### edge_features_np = edge_features.clone().detach().numpy(); # print(type(edge_features_np))  # Should be <class 'numpy.ndarray'> 
        ### edge_weights_np = edge_weights.clone().detach().numpy()

        '''x_np = self.x.clone().detach().numpy(); # print(type(x_np))  # Should be <class 'numpy.ndarray'>
        edge_features_np = instance.edge_features.clone().detach().numpy()
        edge_features_np = edge_features.clone().detach().numpy(); ''' # print(type(edge_features_np))  # Should be <class 'numpy.ndarray'> 

        self.edge_indices = torch.where(self.A_v_bar != 0) # integer tensor
        # print(f"edge_indices.shape: {self.edge_indices}")

        edge_weights = self.A_v_bar[self.edge_indices] # real tensor
        edge_weights_np = edge_weights.clone().detach().numpy()

        '''P_node_sigmoid = torch.sigmoid(self.P_node_hat) # Repeating for P_node_hat
        mask_node = (P_node_sigmoid >= .5).float().clone()
        P_node = P_node_sigmoid + (mask_node - P_node_sigmoid).detach()
        use P_node instead of self.P_node_hat'''
        self.N_v_bar = self.P_node_hat * self.node_features.cpu()
        # self.N_v_bar_copy = torch.tensor(self.N_v_bar.data, requires_grad=True)
        N_v_bar_np = self.N_v_bar.clone().detach().numpy()

        '''try:
            edge_features = instance.edge_features[self.edge_indices]

            A_v_bar_GI = GraphInstance(
            id=instance.id,
            label="CF",
            data = A_v_bar_np,
            node_features = instance.node_features,
            edge_features = edge_features,
            edge_weights = edge_weights_np,
            graph_features = instance.graph_features
            )
        except:
            # print(instance.data.shape, self.A_v_bar.shape)
            # print(instance.edge_features.shape, edge_weights_np.shape)
            # print(max(self.edge_indices[0]), max(self.edge_indices[1]))
            A_v_bar_GI = GraphInstance(
            id=instance.id,
            label="CF",
            data = A_v_bar_np,
            node_features = instance.node_features,
            # edge_features = edge_features,
            edge_weights = edge_weights_np,
            graph_features = instance.graph_features
            )'''


        # if (edge_weights_np.size != edge_features.size): print(edge_weights_np.size, edge_features.size)
        # print(edge_features.size, instance.edge_features.size)
        
        # print(f"edge_weights.shape: {edge_weights.shape}")
        # print(f"edge_weights.dtype: {edge_weights.dtype}")
        # print(f"edge_weights: {edge_weights}")
        # print(f"edge_features: {edge_features}")
                
        # edge_indices = torch.stack(edge_indices).float()

        # print(f"Are P_sigmoid and edge_weights equal: {torch.equal(P_sigmoid, edge_weights)}")

        A_v_bar_GI = GraphInstance(
            id=instance.id,
            label="CF",
            data = A_v_bar_np,
            node_features = N_v_bar_np, # instance.node_features,
            # edge_features = edge_features,
            edge_weights = edge_weights_np,
            graph_features = instance.graph_features
            )
                
        # self.oracle.predict(instance); exit()

        # g_v_bar_pred = self.oracle.predict(A_v_bar_GI) # Oracle CF prediction
        # print(f"Oracle prediction for potential CF: {g_v_bar_pred}")

        def _real_predict_gradients(data_instance):
            '''We added this since we needed gradients to compute how the perturbation on P influences Loss'''
            return _real_predict_proba_gradients(data_instance)

        def _real_predict_proba_gradients(data_inst):
            # print(f"type torch.py ln46: {type(data_inst.data)}")
            # instance_data_np = np.array(data_inst.data) if isinstance(data_inst.data, memoryview) else data_inst.data
            data_inst = to_geometric_gradients(data_inst)
            self.node_features = data_inst.x.to(self.oracle.device)
            # print("node features requires grad?", self.node_features.requires_grad)
            self.edge_index = data_inst.edge_index.to(self.oracle.device)
            self.edge_weights = data_inst.edge_attr.to(self.oracle.device)
            
            return self.oracle.model(self.node_features,self.edge_index,self.edge_weights, None).cpu().squeeze()

        def to_geometric_gradients(instance: GraphInstance, label=0) -> Data:
            self.adj = torch.from_numpy(instance.data).double().requires_grad_(True)
            self.x = torch.from_numpy(instance.node_features).double().requires_grad_(True)
            self.a = torch.nonzero(self.adj).int()
            self.w = torch.from_numpy(instance.edge_weights).double().requires_grad_(True)
            label = torch.tensor(label).long()

            # print(f"edge_weights: {instance.edge_weights}")
            # print(f"a.shape: {self.a.shape}")            


            '''b = torch.nonzero(self.adj).int()

            # 1. Get the integer part and fractional part of the float_tensor
            int_part = torch.floor(self.adj).long()  # Differentiable approximation of index
            frac_part = self.adj - torch.floor(self.adj)  # Fractional part for interpolation
            

            # 2. Split the integer parts into row and column indices
            int_row = int_part[:, 0]#.clamp(0, self.adj.size(0) - 2)  # Row indices
            int_col = int_part[:, 1]#.clamp(0, self.adj.size(1) - 2)  # Column indices

            # 3. Interpolation happens between row and column pairs

            # Example of neighboring row/column for interpolation
            row_left = int_row  # Row indices (left)
            row_right = int_row + 1  # Next row (right neighbor)

            col_left = int_col  # Column indices (left)
            col_right = int_col + 1 # Next column (right neighbor)

            # 4. Perform linear interpolation element-wise for both dimensions

            # Here we interpolate across the row and column fractional parts separately
            # This interpolation is element-wise between the neighboring rows and columns
            interpolated_rows = row_left * (1 - frac_part[:, 0]) + row_right * frac_part[:, 0]
            interpolated_cols = col_left * (1 - frac_part[:, 1]) + col_right * frac_part[:, 1]

            # Combine interpolated row and column results
            interpolated_values = torch.stack((interpolated_rows, interpolated_cols), dim=1)

            # Check if they are equal
            print("Are interpolated values equal to b?", interpolated_values.int().equal(b))
            # print(interpolated_values.int() == b)
            print(interpolated_values)
            print(b)

            print(f"Does interpolated_values requires grads: {interpolated_values.requires_grad}")
            exit()'''
            return Data(x=self.x, y=label, edge_index=self.a.T, edge_attr=self.w)

        # self.g_v_logits = torch.tensor(self.oracle._real_predict_gradients(A_v_bar_GI), requires_grad=True) # Oracle CF prediction → returns probabilities
        self.g_v_logits = _real_predict_gradients(A_v_bar_GI) # Oracle CF prediction → returns probabilities
        g_v_bar_pred = torch.argmax(self.g_v_logits).item()
        # print("g_v_logits type:", type(self.g_v_logits))

        # print(f"g_v_bar_pred: {g_v_bar_pred}")

        valid_CF = False
        self.w_grads = self.w.grad
        with torch.no_grad():
            if self.f_v.item() != g_v_bar_pred: # If valid CF
                valid_CF = True # Valid CF is found
                v_bar = v_bar_cand
                # v_bar = v_bar_cand[0].clone(), v_bar_cand[1]
                if not self.opt_flag:
                    self.v_bar_opt = v_bar#.clone() # First CF
                    print(f"{Color.GREEN}Found valid counterfactual! {Color.CYAN}Counterfactual predicted class: {Color.RESET}{g_v_bar_pred}")  # Debugging
                    self.edge_weights_opt = edge_weights_np
                    self.opt_flag = True
                elif self.__distance(self.v[0], v_bar[0]) < self.__distance(self.v[0], self.v_bar_opt[0]):
                    self.v_bar_opt = v_bar#.clone() # Keep track of best CF
                    self.edge_weights_opt = edge_weights_np
                    print(f"{Color.BLUE}Found new best valid counterfactual! {Color.CYAN}Counterfactual predicted class: {Color.RESET}{g_v_bar_pred}")  # Debugging

        # print(v_bar_cand)#[0].requires_grad())
        return v_bar_cand, valid_CF

    def __calculate_loss(self, v_bar, instance, valid_CF):
        """
        Loss function based on:
        L = Lpred(v,  v_bar | f, g) + βLdist(v, v_bar | d) Eq(1) [4].
        L_pred(v, v_bar | f, g) = -1[f(v) = f(v_bar)] * L_NLL(f(v), g(v_bar)) Eq(5) [5.3].
        L_dist(v, v_bar | d): the element-wise difference between A_v and A_v_bar, i.e., the number of edges removed.
        """
        # A_v, x = v
        # A_v_bar, x_bar = v_bar
        
        # g_v_bar_pred = torch.where(self.f_v==1, 0, 1) if valid_CF else self.f_v

        g_v_probabilities = F.softmax(self.g_v_logits, dim=0)
        self.g_v_probabilities = g_v_probabilities
        f_v_probabilities = torch.nn.functional.one_hot(self.f_v, num_classes=self.dataset.num_classes).long()
        # print(g_v_probabilities); print(f_v_probabilities)


        # 1. Prediction loss: L_pred
        # Get the original prediction and counterfactual prediction
        
        # If f(v) == f(v_bar), the loss is 0; otherwise, compute NLL loss
        if not valid_CF: 
            # Use negative log-likelihood loss (NLL) between for predicted logits and ground truth (Loss Function Optimization) [5.3]
            L_pred = self.loss_fn(g_v_probabilities, f_v_probabilities)
            L_pred = -1 * L_pred
        else:
            L_pred = 0  # No loss if predictions are different
        
        # 2. Distance loss: L_dist
        # Compute the number of edges removed (element-wise difference between A_v and A_v_bar)
        edge_diff = torch.abs(self.A_v - self.A_v_bar)
        node_diff = torch.abs(self.node_features.cpu() - self.N_v_bar)
        # edge_diff = torch.abs(self.A_v - v_bar[0]) # Trying for debugging but not correct
        D_edges = edge_diff.sum() # Count the number of edges changed
        D_nodes = node_diff.sum() # Distance between the nodes changed
        L_dist = D_edges + D_nodes
        # print(f"edge_diff.sum(): {edge_diff.sum()}")
        # print(f"node_diff.sum(): {node_diff.sum()}")
        
        # 3. Total loss
        total_loss = L_pred + self.β * L_dist
        
        # return L_pred
        return total_loss
    
    @torch.no_grad()
    def __distance(self, v:GraphInstance, CF:GraphInstance):
        
        v = v.clone().detach()
        CF = CF.clone().detach()

        l2norm = torch.norm(v-CF, p=2) # l2-norm of the difference

        return l2norm
    

    def check_configuration(self):
        super().check_configuration()
        local_config = self.local_config
        
        if 'overshoot_factor' not in local_config['parameters']:
            local_config['parameters']['overshoot_factor'] = 2
        
        if 'step_size' not in local_config['parameters']:
            local_config['parameters']['step_size'] = 0.2
            
        if 'max_iterations' not in local_config['parameters']:
            local_config['parameters']['max_iterations'] = 10
            
        if 'perturbation' not in local_config['parameters']:
            local_config['parameters']['perturbation'] = 1
            
        if 'threshold' not in local_config['parameters']:
            local_config['parameters']['threshold'] = 0.3
               
        self.fold_id = self.local_config['parameters'].get('fold_id',-1)

class Color: # Print nice console colors for readibility' sake
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    RESET = '\033[0m'  