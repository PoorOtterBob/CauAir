import time
import numpy as np
import random
from torch.nn import functional as F
import torch
import torch.nn as nn
from torchdiffeq import odeint
from torch.nn.modules.rnn import GRU
from src.base.model import BaseModel
from src.utils.module.airphynet.utils import *


'''def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)'''

'''class EncoderAttrs:
    def __init__(self, adj_mx, **model_kwargs):
        self.adj_mx = adj_mx
        self.num_nodes = adj_mx.shape[0]
        self.num_edges = (adj_mx > 0.).sum()
        self.gcn_step = int(model_kwargs.get('gcn_step', 2))
        self.filter_type = model_kwargs.get('filter_type')
        self.num_rnn_layers = int(model_kwargs.get('num_rnn_layers', 1))
        self.rnn_units = int(model_kwargs.get('rnn_units'))
        self.latent_dim = int(model_kwargs.get('latent_dim', 4))'''

class AirPhyNetModel(BaseModel):
    def __init__(self, 
                 adj_mx, 
                 edge_index, 
                 edge_attr, 
                 logger, 
                 device, 
                 **args):
        super(AirPhyNetModel, self).__init__(**args)
        self._logger = logger

        self.latent_dim = 4
        self.n_traj_samples = 3 # int(model_kwargs.get('n_traj_samples', 1))
        self.ode_method = 'dopri5' # model_kwargs.get('ode_method', 'dopri5')
        self.atol = 0.00001 # float(model_kwargs.get('odeint_atol', 1e-4))
        self.rtol = 0.00001 # float(model_kwargs.get('odeint_rtol', 1e-3))
        self.num_gen_layer = 1 # int(model_kwargs.get('gen_layers', 1))
        self.ode_gen_dim = 64 # int(model_kwargs.get('gen_dim', 64))

        self.gcn_step = 2
        self.filter_type = 'diff'

        self.adj_mx =  adj_mx
        self.edge_index =  edge_index
        self.edge_attr =  edge_attr

        ####################################################
        # RNN Encoder
        ####################################################
        self.encoder_z0 = Encoder_z0_RNN(adj_mx, 
                                         self.input_dim, 
                                         rnn_units=64, 
                                         latent_dim=4, 
                                         node_num=self.node_num, 
                                         device=device)

        ####################################################
        # ODE solver
        ####################################################
        ode_set_str = "ODE setting --latent {} --samples {} --method {} \
            --atol {:6f} --rtol {:6f} --gen_layer {} --gen_dim {}".format(\
                self.latent_dim, self.n_traj_samples, self.ode_method, \
                self.atol, self.rtol, self.num_gen_layer, self.ode_gen_dim)
        
        
        

        self._logger.info(ode_set_str)

        self.save_latent = False # bool(model_kwargs.get('save_latent', False))
        self.latent_feat = None # used to extract the latent feature

        ####################################################
        # Decoder
        ####################################################
        self.horizon = self.horizon # int(model_kwargs.get('horizon', 1))
        self.out_feat = self.output_dim # int(model_kwargs.get('output_dim', 1))
        self.decoder = Decoder(
            self.out_feat,
            adj_mx,
            self.node_num, # self.num_nodes,
            (adj_mx > 0.).sum(), # max(edge_index.shape) # self.num_edges,
        ).to(device)
        self.device = device

    ##########################################
    def forward(self, inputs, labels=None, batches_seen=None):
        # b t n f -> t b n*f
        bs = inputs.shape[0]
        inputs = inputs.transpose(0, 1).reshape(self.seq_len, -1, self.input_dim)
        """
        seq2seq forward pass
        :param inputs: shape (seq_len, batch_size, num_nodes * input_dim)
        :param labels: shape (horizon, batch_size, num_nodes * output_dim)
        :param batches_seen: batches seen till now
        :return: outputs: (self.horizon, batch_size, self.num_edges * self.output_dim)
        """
        perf_time = time.time()
        # shape: [1, batch, num_nodes * latent_dim]
        first_point_mu, first_point_std, last_wind_vars = self.encoder_z0(inputs, bs)
        self._logger.debug("Recognition complete with {:.1f}s".format(time.time() - perf_time))

        # sample 'n_traj_samples' trajectory
        perf_time = time.time()
        means_z0 = first_point_mu.repeat(self.n_traj_samples, 1, 1)
        sigma_z0 = first_point_std.repeat(self.n_traj_samples, 1, 1)
        first_point_enc = sample_standard_gaussian(means_z0, sigma_z0)

        time_steps_to_predict = torch.arange(start=0, end=self.horizon, step=1).float().to(self.device)
        time_steps_to_predict = time_steps_to_predict / len(time_steps_to_predict)

        # Shape of sol_ys (horizon, n_traj_samples, batch_size, self.num_nodes * self.latent_dim)
        odefunc = ODEFunc(last_wind_vars, self.ode_gen_dim, self.latent_dim, self.adj_mx, self.edge_index, self.edge_attr,
                          self.gcn_step, self.node_num, filter_type=self.filter_type, device=self.device).to(self.device)
        self.diffeq_solver = DiffeqSolver(odefunc,self.ode_method, self.latent_dim, odeint_rtol=self.rtol,
                                          odeint_atol=self.atol)
        sol_ys, fe = self.diffeq_solver(first_point_enc, time_steps_to_predict)
        '''self._logger.debug("ODE solver complete with {:.1f}s".format(time.time() - perf_time))'''
        '''if(self.save_latent):
            # Shape of latent_feat (horizon, batch_size, self.num_nodes * self.latent_dim)
            self.latent_feat = torch.mean(sol_ys.detach(), axis=1)'''

        '''perf_time = time.time()
        
        self._logger.debug("Decoder complete with {:.1f}s".format(time.time() - perf_time))'''

        '''if batches_seen == 0:
            self._logger.info(
                "Total trainable parameters {}".format(count_parameters(self))
            )'''
        outputs = self.decoder(sol_ys)
        outputs = outputs.transpose(0,  1).reshape(-1, self.seq_len, self.node_num, self.output_dim)
        # return outputs, fe
        return outputs

class Encoder_z0_RNN(nn.Module): # , EncoderAttrs):
    def __init__(self, adj_mx, input_dim, rnn_units, latent_dim, node_num, device):
        nn.Module.__init__(self)
        # EncoderAttrs.__init__(self, get_adjacency_matrix)
        self.recg_type = 'gru' # model_kwargs.get('recg_type', 'gru') # gru
        self.rnn_units = rnn_units
        self.latent_dim = latent_dim
        self.num_nodes = node_num 
        if(self.recg_type == 'gru'):
            # gru settings
            self.input_var = input_dim, # int(model_kwargs.get('input_var', 3))
            self.input_dim = 1 # int(model_kwargs.get('input_dim', 1))
            self.gru_rnn = GRU(self.input_dim, self.rnn_units).to(device)
        else:
            raise NotImplementedError("The recognition net only support 'gru'.")

        # hidden to z0 settings
        self.hiddens_to_z0 = nn.Sequential(
            nn.Linear(self.rnn_units, 50),
            nn.Tanh(),
            nn.Linear(50, self.latent_dim * 2),)

        init_network_weights(self.hiddens_to_z0)

    def forward(self, inputs, batch_size):
        """
        encoder forward pass on t time steps
        :param inputs: shape (seq_len, batch_size, num_edges * input_var)
        :return: mean, std: # shape (n_samples=1, batch_size, self.latent_dim)
        """
        if(self.recg_type == 'gru'):
            # shape of outputs: (seq_len, batch, num_senor * rnn_units)
            # seq_len, batch_size = inputs.shape[0], inputs.shape[1]
            # inputs = inputs.reshape(seq_len, batch_size, self.num_nodes, self.input_var)
            # inputs = inputs.reshape(seq_len, batch_size * self.num_nodes, self.input_var) #(24, 1120, 6)

            pm25 = inputs[:,:,0].unsqueeze(-1)
            wind_vars = inputs[:,:,-2:] 
            outputs, _ = self.gru_rnn(pm25) \

            last_output = outputs[-1]
            last_output = torch.reshape(last_output, (batch_size, self.num_nodes, -1))  # (batch_size, num_nodes, rnn_units) 
            last_wind_vars = torch.reshape(wind_vars[-1], (batch_size, self.num_nodes, -1)) #(batch_size, num_nodes, wind_dim) 
        else:
            raise NotImplementedError("The recognition net only support 'gru'.")

        mean, std = split_last_dim(self.hiddens_to_z0(last_output))
        mean = mean.reshape(batch_size, -1) # (batch_size, num_nodes * latent_dim)(32, 140)
        std = std.reshape(batch_size, -1) # (batch_size, num_nodes * latent_dim)(32, 140)
        std = std.abs()

        assert(not torch.isnan(mean).any())
        assert(not torch.isnan(std).any())

        return mean.unsqueeze(0), std.unsqueeze(0), last_wind_vars # for n_sample traj

class Decoder(nn.Module):
    def __init__(self, output_dim, adj_mx, num_nodes, num_edges):
        super(Decoder, self).__init__()

        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.output_dim = output_dim

    def forward(self, inputs):
        """
        :param inputs: (horizon, n_traj_samples, batch_size, num_nodes * latent_dim)
        :return outputs: (horizon, batch_size, num_nodes * output_dim), average result of n_traj_samples.
        """
        assert(len(inputs.size()) == 4)
        horizon, n_traj_samples, batch_size = inputs.size()[:3]

        inputs = inputs.reshape(horizon, n_traj_samples, batch_size, self.num_nodes, -1).transpose(-2, -1)
        latent_dim = inputs.size(-2)
        outputs = inputs.reshape(horizon, n_traj_samples, batch_size, latent_dim, self.num_nodes, self.output_dim)

        outputs = torch.mean(
            torch.mean(outputs, axis=3),
            axis=1
        )
        outputs = outputs.reshape(horizon, batch_size, -1)
        return outputs

class LayerParams:
    def __init__(self, rnn_network: nn.Module, layer_type: str, device):
        self._rnn_network = rnn_network
        self._params_dict = {}
        self._biases_dict = {}
        self._type = layer_type
        self._device = device
    def get_weights(self, shape):
        if shape not in self._params_dict:
            nn_param = nn.Parameter(torch.empty(*shape, device=self._device))
            nn.init.xavier_normal_(nn_param)
            self._params_dict[shape] = nn_param
            self._rnn_network.register_parameter('{}_weight_{}'.format(self._type, str(shape)),
                                                 nn_param)
        return self._params_dict[shape]

    def get_biases(self, length, bias_start=0.0):
        if length not in self._biases_dict:
            biases = nn.Parameter(torch.empty(length, device=self._device))
            nn.init.constant_(biases, bias_start)
            self._biases_dict[length] = biases
            self._rnn_network.register_parameter('{}_biases_{}'.format(self._type, str(length)),
                                                 biases)

        return self._biases_dict[length]

class LinearNet(nn.Module):
    def __init__(self):
        super(LinearNet, self).__init__()
        self.fc = nn.Linear(2, 1)

    def forward(self, x):
        x = self.fc(x)
        return x


class GatedFusionModel(torch.nn.Module):
    def __init__(self, num_nodes,latent_dim):
        super(GatedFusionModel, self).__init__()
        self._num_nodes = num_nodes
        self._latent_dim = latent_dim
        self.hid_dim = self._num_nodes*self._latent_dim
        self.fc = nn.Linear(self.hid_dim,self.hid_dim)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, grad_diff, grad_adv):
        X_diff = self.fc(grad_diff)
        X_adv = self.fc(grad_adv)
        z = self.sigmoid(torch.add(X_diff,X_adv))
        H = torch.add((z * X_diff), ((1 - z) * X_adv))
        return H

class ODEFunc(nn.Module):
    def __init__(self, 
                 last_wind_vars, 
                 num_units, 
                 latent_dim, 
                 adj_mx, 
                 edge_index, 
                 edge_attr, 
                 gcn_step, 
                 num_nodes,
                 gen_layers=1, nonlinearity='tanh', filter_type="diff_adv", device=None):
        """
        :param num_units: dimensionality of the hidden layers
        :param latent_dim: dimensionality used for ODE (input and output). Analog of a continous latent state
        :param adj_mx:
        :param gcn_step:
        :param num_nodes:
        :param gen_layers: hidden layers in each ode func.
        :param nonlinearity:
        """
        super(ODEFunc, self).__init__()
        self._activation = torch.tanh if nonlinearity == 'tanh' else torch.relu

        self._num_nodes = num_nodes
        self._num_units = num_units 
        self._latent_dim = latent_dim
        self._gen_layers = gen_layers
        self.nfe = 0 #Number of function integrations
        self.flow_net = LinearNet().to(device)
        self.gated_fusion = GatedFusionModel(self._num_nodes, self._latent_dim)

        self._filter_type = filter_type
        self._device = device
        
        if(self._filter_type == "diff"):
            self._gcn_step = gcn_step
            self._gconv_params = LayerParams(self, 'gconv', device=device)
            self._supports = []
            supports = []
            supports.append(calculate_scaled_laplacian(adj_mx))

            for support in supports:
                self._supports.append(self._build_sparse_matrix(support, device))

        '''elif(self._filter_type == "adv"):
            edge_index = torch.LongTensor(edge_index).to(device) 
            edge_attr = torch.Tensor(np.float32(edge_attr)).to(device) 
            edge_src, edge_target = edge_index

            last_wind_vars = self.flow_net(last_wind_vars) 
            node_src = last_wind_vars[:,edge_src] 
            node_target = last_wind_vars[:,edge_target]
            edge_weight = node_src - node_target
            edge_weight = edge_weight.squeeze()

            batch_size, num_edges =  edge_weight.shape
            adj_mx_adv = torch.zeros(batch_size, num_nodes, num_nodes)
            for batch_index in range(batch_size):
                for edge_id in range(num_edges):
                    src_node = edge_index[0, edge_id]
                    tgt_node = edge_index[1, edge_id]

                    # Assign the edge weight to the adjacency matrix
                    adj_mx_adv[batch_index,src_node, tgt_node] = edge_weight[batch_index,edge_id]

            self._gcn_step = gcn_step
            self._gconv_adv_params = LayerParams(self, 'gconv_adv')

            #For Advection
            self._supports_adv = []
            supports_adv = []
            for i in range(batch_size):
                adj_mx_new  = adj_mx_adv[i]
                supports_adv.append(calculate_scaled_laplacian(adj_mx_new.detach().numpy()))
            for support in supports_adv:
                self._supports_adv.append(self._build_sparse_matrix(support, device))

        elif(self._filter_type == "diff_adv"):
            edge_index = torch.LongTensor(edge_index).to(device) 
            edge_attr = torch.Tensor(np.float32(edge_attr)).to(device) 
            edge_src, edge_target = edge_index

            last_wind_vars = self.flow_net(last_wind_vars) 
            node_src = last_wind_vars[:,edge_src]
            node_target = last_wind_vars[:,edge_target]
            edge_weight = node_src - node_target
            edge_weight = edge_weight.squeeze() 

            batch_size, num_edges =  edge_weight.shape
            adj_mx_adv = torch.zeros(batch_size, num_nodes, num_nodes)
            for batch_index in range(batch_size):
                for edge_id in range(num_edges):
                    src_node = edge_index[0, edge_id]
                    tgt_node = edge_index[1, edge_id]

                    # Assign the edge weight to the adjacency matrix
                    adj_mx_adv[batch_index,src_node, tgt_node] = edge_weight[batch_index,edge_id]

            self._gcn_step = gcn_step
            self._gconv_params = LayerParams(self, 'gconv')
            self._gconv_adv_params = LayerParams(self, 'gconv_adv')

            #For Advection
            self._supports_adv = []
            supports_adv = []
            for i in range(batch_size):
                adj_mx_new  = adj_mx_adv[i]
                supports_adv.append(calculate_scaled_laplacian(adj_mx_new.detach().numpy()))
            for support in supports_adv:
                self._supports_adv.append(self._build_sparse_matrix(support, device))

            #For Diffusion
            self._supports = []
            supports = []
            supports.append(calculate_scaled_laplacian(adj_mx))
            for support in supports:
                self._supports.append(self._build_sparse_matrix(support, device))
            
        elif(self._filter_type == "unkP"):
            ode_func_net = create_net(latent_dim, latent_dim, n_units = num_units)
            init_network_weights(ode_func_net)
            self.gradient_net = ode_func_net
            
        else:
            print("Invalid Filter Type")'''


    @staticmethod
    def _build_sparse_matrix(L, device):
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        # this is to ensure row-major ordering to equal torch.sparse.sparse_reorder(L)
        indices = indices[np.lexsort((indices[:, 0], indices[:, 1]))]
        L = torch.sparse_coo_tensor(indices.T, L.data, L.shape, device=device)
        return L

    def forward(self, t_local, y, backwards = False):
        """
        t_local: current time point
        y: value at the current time point, shape (B, num_nodes * latent_dim)
        Output: (B, num_nodes * latent_dim)`.
        """
        self.nfe += 1
        grad = self.get_ode_gradient_nn(t_local, y)
        if backwards:
            grad = -grad
        return grad

    def get_ode_gradient_nn(self, t_local, inputs):
        coeff = 0.1
        if(self._filter_type == "diff"):
            grad = - coeff * self.ode_func_net_diff(inputs, self._supports)
        elif(self._filter_type == "adv"):
            grad = - self.ode_func_net_adv(inputs, self._supports_adv)
        elif(self._filter_type == "diff_adv"):
            grad_diff = - coeff * self.ode_func_net_diff(inputs, self._supports)
            grad_adv = - self.ode_func_net_adv(inputs, self._supports_adv)
            grad = self.gated_fusion(grad_diff, grad_adv)
        elif(self._filter_type == "unkP"):
            grad = self._fc(inputs)
        else:
            print("Invalid Filter Type")

        return grad

    def ode_func_net_diff(self, inputs, _supports):
        c = inputs
        for i in range(self._gen_layers):
            c = self._gconv_dif(c, self._num_units,_supports)
            c = self._activation(c)
        c = self._gconv_dif(c, self._latent_dim,_supports)
        c = self._activation(c)
        return c

    def ode_func_net_adv(self, inputs,_supports_adv):
        c = inputs
        for i in range(self._gen_layers):
            c = self._gconv_adv(c, self._num_units,_supports_adv)
            c = self._activation(c)
        c = self._gconv_adv(c, self._latent_dim,_supports_adv)
        c = self._activation(c)
        return c


    def _fc(self, inputs):
        batch_size = inputs.size()[0]
        grad = self.gradient_net(inputs.view(batch_size * self._num_nodes, self._latent_dim))
        return grad.reshape(batch_size, self._num_nodes * self._latent_dim) # (batch_size, num_nodes, latent_dim)

    @staticmethod
    def _concat(x, x_):
        x_ = x_.unsqueeze(0)
        return torch.cat([x, x_], dim=0)

    def _gconv_dif(self, inputs, output_size,_supports, bias_start=0.0):
        # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim)
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self._num_nodes, -1))
        input_size = inputs.size(2)

        x = inputs
        x0 = x.permute(1, 2, 0)  # (num_nodes, total_arg_size, batch_size)
        x0 = torch.reshape(x0, shape=[self._num_nodes, input_size * batch_size])
        x = torch.unsqueeze(x0, 0)

        if self._gcn_step == 0:
            pass
        else:
            for support in self._supports:
                x1 = torch.sparse.mm(support, x0)
                x = self._concat(x, x1)

                for k in range(2, self._gcn_step + 1):
                    x2 = 2 * torch.sparse.mm(support, x1) - x0
                    x = self._concat(x, x2)
                    x1, x0 = x2, x1

        num_matrices = len(self._supports) * self._gcn_step + 1  # Adds for x itself.
        x = torch.reshape(x, shape=[num_matrices, self._num_nodes, input_size, batch_size])
        x = x.permute(3, 1, 2, 0)  # (batch_size, num_nodes, input_size, order)
        x = torch.reshape(x, shape=[batch_size * self._num_nodes, input_size * num_matrices])

        weights = self._gconv_params.get_weights((input_size * num_matrices, output_size))
        x = torch.matmul(x, weights)  # (batch_size * self._num_nodes, output_size)

        biases = self._gconv_params.get_biases(output_size, bias_start)
        x += biases
        # Reshape res back to 2D: (batch_size, num_node, state_dim) -> (batch_size, num_node * state_dim)
        return torch.reshape(x, [batch_size, self._num_nodes * output_size])

    def _gconv_adv(self, inputs, output_size,_supports_adv, bias_start=0.0):
        # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim)
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self._num_nodes, -1))
        input_size = inputs.size(2)

        x = inputs
        x0 = x.permute(1, 2, 0)  # (num_nodes, total_arg_size, batch_size)
        x0 = torch.reshape(x0, shape=[self._num_nodes, input_size * batch_size])
        x = torch.unsqueeze(x0, 0)

        if self._gcn_step == 0:
            pass
        else:
            for support in self._supports_adv:
                x1 = torch.sparse.mm(support, x0)
                x = self._concat(x, x1)

                for k in range(2, self._gcn_step + 1):
                    x2 = 2 * torch.sparse.mm(support, x1) - x0
                    x = self._concat(x, x2)
                    x1, x0 = x2, x1

        num_matrices = len(self._supports_adv) * self._gcn_step + 1  # Adds for x itself.
        x = torch.reshape(x, shape=[num_matrices, self._num_nodes, input_size, batch_size])
        x = x.permute(3, 1, 2, 0)  # (batch_size, num_nodes, input_size, order)
        x = torch.reshape(x, shape=[batch_size * self._num_nodes, input_size * num_matrices])

        weights = self._gconv_adv_params.get_weights((input_size * num_matrices, output_size))
        x = torch.matmul(x, weights)  # (batch_size * self._num_nodes, output_size)

        biases = self._gconv_adv_params.get_biases(output_size, bias_start)
        x += biases
        # Reshape res back to 2D: (batch_size, num_node, state_dim) -> (batch_size, num_node * state_dim)
        return torch.reshape(x, [batch_size, self._num_nodes * output_size])
    
class DiffeqSolver(nn.Module):
    def __init__(self, odefunc, method, latent_dim,
            odeint_rtol = 1e-4, odeint_atol = 1e-5):
        nn.Module.__init__(self)

        self.ode_method = method
        self.odefunc = odefunc
        self.latent_dim = latent_dim

        self.rtol = odeint_rtol
        self.atol = odeint_atol

    def forward(self, first_point, time_steps_to_pred):
        """
        Decoder the trajectory through the ODE Solver.
        :param time_steps_to_pred: horizon
        :param first_point: (n_traj_samples, batch_size, num_nodes * latent_dim)
        :return: pred_y: # shape (horizon, n_traj_samples, batch_size, self.num_nodes * self.output_dim)
        """
        n_traj_samples, batch_size = first_point.size()[0], first_point.size()[1]
        first_point = first_point.reshape(n_traj_samples * batch_size, -1) # reduce the complexity by merging dimension

        # pred_y shape: (horizon, n_traj_samples * batch_size, num_nodes * latent_dim)
        start_time = time.time()
        self.odefunc.nfe = 0
        pred_y = odeint(self.odefunc,
                            first_point,
                            time_steps_to_pred,
                            rtol=self.rtol,
                            atol=self.atol,
                            method=self.ode_method)
        time_fe = time.time() - start_time

        # pred_y shape: (horizon, n_traj_samples, batch_size, num_nodes * latent_dim)
        pred_y = pred_y.reshape(pred_y.size()[0], n_traj_samples, batch_size, -1)

        return pred_y, (self.odefunc.nfe, time_fe)