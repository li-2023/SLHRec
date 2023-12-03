import torch
import torch.nn as nn
from cmd_args import cmd_args
from predicate import PRED_DICT
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import random
from torch_geometric.nn import RGCNConv, APPNP, GCNConv, RGATConv


class GATEITU(nn.Module):
    
    def __init__(self, emb_size):
        super(GATEITU, self).__init__()
        
        self.emb_size = emb_size
        self.tm1 = nn.Linear(self.emb_size, self.emb_size)
        self.tm2 = nn.Linear(self.emb_size, self.emb_size)
        self.gate1 = nn.Linear(2*self.emb_size, self.emb_size)
        self.gate2 = nn.Linear(2*self.emb_size, self.emb_size)
        self.sigmoid = nn.Sigmoid()
        
        nn.init.xavier_uniform_(self.tm1.weight)
        nn.init.xavier_uniform_(self.tm2.weight)
        nn.init.xavier_uniform_(self.gate1.weight)
        nn.init.xavier_uniform_(self.gate2.weight)

    def forward(self, emb1, emb2):
        
        emb2_t = self.tm1(emb2)
        emb1_t = self.tm2(emb1)
        
        cat_vec1 = torch.cat([emb1, emb2_t], dim=1)
        cat_vec2 = torch.cat([emb2, emb1_t], dim=1)
        
        gate1 = self.sigmoid(self.gate1(cat_vec1))
        gate2 = self.sigmoid(self.gate2(cat_vec2))
        
        gate_vec1 = gate1*emb2_t
        gate_vec2 = gate2*emb1_t
        
        final_emb1 = gate_vec1 + emb1
        final_emb2 = gate_vec2 + emb2
        
        return final_emb1, final_emb2
    
    
class ATTITU(nn.Module):
    
    def __init__(self, emb_size):
        super(ATTITU, self).__init__()
        
        self.emb_size = emb_size
        self.att_linear11 = nn.Linear(self.emb_size, 1)
        self.att_linear12 = nn.Linear(self.emb_size, 1)
        self.att_linear21 = nn.Linear(self.emb_size, 1)
        self.att_linear22 = nn.Linear(self.emb_size, 1)

        nn.init.xavier_uniform_(self.att_linear11.weight)
        nn.init.xavier_uniform_(self.att_linear12.weight)
        nn.init.xavier_uniform_(self.att_linear21.weight)
        nn.init.xavier_uniform_(self.att_linear22.weight)

        
    def forward(self, emb1, emb2):
        emb1_1 = self.att_linear11(emb1)
        emb2_1 = self.att_linear12(emb2)
        emb1_2 = self.att_linear21(emb1)
        emb2_2 = self.att_linear22(emb2)
        
        score1 = F.softmax(torch.cat([emb1_1, emb2_1], dim=1), dim=1)
        final_emb1_self = torch.index_select(score1, 1, torch.LongTensor([0])) * emb1
        final_emb1_side = torch.index_select(score1, 1, torch.LongTensor([1])) * emb2
        final_emb1 = final_emb1_self + final_emb1_side

        score2 = F.softmax(torch.cat([emb1_2, emb2_2], dim=1), dim=1)
        final_emb2_self = torch.index_select(score2, 1, torch.LongTensor([1])) * emb2
        final_emb2_side = torch.index_select(score2, 1, torch.LongTensor([0])) * emb1
        final_emb2 = final_emb2_self + final_emb2_side

        return final_emb1, final_emb2


class GNN_passing(torch.nn.Module):
    def __init__(self, num_node, latent_dim, num_relation_ex, num_relation_im, GNN_type, trans_type, agg_layer_type):
        super().__init__()

        self.GNN_type = GNN_type
        self.trans_type = trans_type
        self.agg_layer_type = agg_layer_type

        if self.GNN_type == "RGCN":
          self.ex_conv1 = RGCNConv(num_node, latent_dim, num_relation_ex)
          self.ex_conv2 = RGCNConv(latent_dim, latent_dim, num_relation_ex)
          self.im_conv1 = RGCNConv(num_node, latent_dim, num_relation_im)
          self.im_conv2 = RGCNConv(latent_dim, latent_dim, num_relation_im)
        elif self.GNN_type == "RGAT":
          self.emb_ex = nn.Parameter(torch.FloatTensor(num_node, latent_dim))
          self.emb_im = nn.Parameter(torch.FloatTensor(num_node, latent_dim))
          nn.init.xavier_uniform_(self.emb_ex)
          nn.init.xavier_uniform_(self.emb_im)
          self.ex_conv1 = RGATConv(num_node, latent_dim, num_relation_ex)
          self.ex_conv2 = RGATConv(latent_dim, latent_dim, num_relation_ex)
          self.im_conv1 = RGATConv(num_node, latent_dim, num_relation_im)
          self.im_conv2 = RGATConv(latent_dim, latent_dim, num_relation_im)
        elif self.GNN_type == "GCN":
          self.emb_ex = nn.Parameter(torch.FloatTensor(num_node, latent_dim))
          self.emb_im = nn.Parameter(torch.FloatTensor(num_node, latent_dim))
          nn.init.xavier_uniform_(self.emb_ex)
          nn.init.xavier_uniform_(self.emb_im)
          self.ex_conv1 = GCNConv(latent_dim, latent_dim)
          self.ex_conv2 = GCNConv(latent_dim, latent_dim)
          self.im_conv1 = GCNConv(latent_dim, latent_dim)
          self.im_conv2 = GCNConv(latent_dim, latent_dim)
        elif self.GNN_type == "APPNP":
          self.emb_ex = nn.Parameter(torch.FloatTensor(num_node, latent_dim))
          self.emb_im = nn.Parameter(torch.FloatTensor(num_node, latent_dim))
          nn.init.xavier_uniform_(self.emb_ex)
          nn.init.xavier_uniform_(self.emb_im)
          self.ex_conv1 = APPNP(1, 0.5)
          self.ex_conv2 = APPNP(1, 0.5)
          self.im_conv1 = APPNP(1, 0.5)
          self.im_conv2 = APPNP(1, 0.5)
        else:
          raise Exception("unknown GNN_type mode: " + self.GNN_type)

        if self.trans_type == "GATEITU":
          self.trans_module = GATEITU(latent_dim)
        elif self.trans_type == "ATTITU":
          self.trans_module = ATTITU(latent_dim)
        else:
          raise Exception("unknown trans_type mode: " + self.trans_type)

        if self.agg_layer_type == 'mlp':
          self.agg_layer1 = nn.Linear(2*latent_dim, latent_dim)
          self.agg_layer2 = nn.Linear(latent_dim, latent_dim)
          nn.init.xavier_uniform_(self.agg_layer1.weight)
          nn.init.xavier_uniform_(self.agg_layer2.weight)

          self.leakyrelu = nn.LeakyReLU()


    def forward(self, edge_index_ex, edge_type_ex, edge_index_im, edge_type_im):
      if self.GNN_type == "RGCN":
        ex_x1 = self.ex_conv1(None, edge_index_ex, edge_type_ex)
        im_x1 = self.im_conv1(None, edge_index_im, edge_type_im)
        t_ex_x1, t_im_x1 = self.trans_module(ex_x1, im_x1)
        ex_x2 = self.ex_conv2(t_ex_x1, edge_index_ex, edge_type_ex)
        im_x2 = self.im_conv2(t_im_x1, edge_index_im, edge_type_im)
      elif self.GNN_type == "RGAT":
        ex_x1 = self.ex_conv1(self.emb_ex, edge_index_ex, edge_type_ex)
        im_x1 = self.im_conv1(self.emb_im, edge_index_im, edge_type_im)
        t_ex_x1, t_im_x1 = self.trans_module(ex_x1, im_x1)
        ex_x2 = self.ex_conv2(t_ex_x1, edge_index_ex, edge_type_ex)
        im_x2 = self.im_conv2(t_im_x1, edge_index_im, edge_type_im)
      elif self.GNN_type == "APPNP":
        ex_x1 = self.ex_conv1(self.emb_ex, edge_index_ex)
        im_x1 = self.im_conv1(self.emb_im, edge_index_im)
        t_ex_x1, t_im_x1 = self.trans_module(ex_x1, im_x1)
        ex_x2 = self.ex_conv2(t_ex_x1, edge_index_ex)
        im_x2 = self.im_conv2(t_im_x1, edge_index_im)
      elif self.GNN_type == "GCN":
        ex_x1 = self.ex_conv1(self.emb_ex, edge_index_ex)
        im_x1 = self.im_conv1(self.emb_im, edge_index_im)
        t_ex_x1, t_im_x1 = self.trans_module(ex_x1, im_x1)
        ex_x2 = self.ex_conv2(t_ex_x1, edge_index_ex)
        im_x2 = self.im_conv2(t_im_x1, edge_index_im)
      else:
          raise Exception("unknown GNN_type mode: " + self.GNN_type)
      
      if self.agg_layer_type == 'mlp':
        cat_emb = torch.cat([ex_x1 + ex_x2, im_x1 + im_x2], dim=1)
        x = self.agg_layer2(self.leakyrelu(self.agg_layer1(cat_emb)))
      elif self.agg_layer_type == 'sum':
        x = ex_x1 + ex_x2 + im_x1 + im_x2
      else:
          raise Exception("unknown agg_layer_type mode: " + self.agg_layer_type)

      return x



class SLHR(nn.Module):
  def __init__(self, graph, latent_dim, num_relation_ex, num_relation_im):
    super(SLHR, self).__init__()

    self.graph = graph
    self.latent_dim = latent_dim

    self.xent_loss = F.binary_cross_entropy_with_logits

    self.device = cmd_args.device

    self.num_ents = graph.num_ents
    self.num_rels = graph.num_rels
    self.ent2idx = graph.ent2idx
    self.rel2idx = graph.rel2idx
    self.idx2rel = graph.idx2rel

    self.r = cmd_args.r
    self.t = cmd_args.t
    self.tau = cmd_args.tau

    self.node_embedding = Parameter(torch.FloatTensor(self.num_ents, self.latent_dim))
    self.node_embedding.data = (1e-3 * torch.randn((self.num_ents, self.latent_dim), dtype=torch.float))

    self.relation_embedding = Parameter(torch.FloatTensor(self.num_rels, self.latent_dim))
    self.relation_embedding.data = (1e-3 * torch.randn((self.num_rels, self.latent_dim), dtype=torch.float))

    self.pre_linear1 = torch.nn.Linear(2 * self.latent_dim, self.latent_dim, bias=True)
    self.pre_linear2 = torch.nn.Linear(self.latent_dim, 1, bias=True)
    
    self.num_relation_ex = num_relation_ex
    self.num_relation_im = num_relation_im
    self.GNN_type = cmd_args.GNN_type
    self.trans_type = cmd_args.trans_type
    self.agg_layer_type = cmd_args.agg_layer_type

    self.GNN_model = GNN_passing(self.num_ents, self.latent_dim, self.num_relation_ex, self.num_relation_im, self.GNN_type, self.trans_type, self.agg_layer_type)
    
  
  def get_struc_embedding(self, edge_index_ex, edge_type_ex, edge_index_im, edge_type_im):

    return self.GNN_model(edge_index_ex, edge_type_ex, edge_index_im, edge_type_im)

  def get_rating(self, u_indices, i_indices, final_embedding):
      u_indices = u_indices.long()
      i_indices = i_indices.long()
      user_emb_s = final_embedding[u_indices]
      item_emb_s = final_embedding[i_indices]
      
      pre_rating = torch.sigmoid(torch.sum(user_emb_s * item_emb_s, 1, keepdim=True))

      return pre_rating

  def get_test_rating(self, u_indices, i_indices, final_embedding, like_str, pre_alpha):
      u_indices = u_indices.long()
      i_indices = i_indices.long()
      
      rel_idx = self.rel2idx[like_str]

      u = u_indices.view((-1, 1))
      v = i_indices.view((-1, 1))

      head_query = self.node_embedding[u]
      tail_query = self.node_embedding[v]

      dist = torch.sum(abs((head_query + self.relation_embedding[rel_idx]) - (self.relation_embedding[rel_idx] * tail_query)), 2)
      score_logic = (1. / (torch.exp((dist - self.r) / self.t) + 1.0))# .sum(dim = 1)

      score_struc = self.get_rating(u_indices, i_indices, final_embedding)

      return (pre_alpha * score_struc) + (1.0 - pre_alpha) * score_logic

  def cal_ssl_loss(self, cl_list, final_embedding):

      cl = cl_list.long()

      struc_cl_embeddings = final_embedding[cl]
      logic_cl_embeddings = self.node_embedding[cl]

      struc_cl_embeddings = F.normalize(struc_cl_embeddings, dim=1)
      logic_cl_embeddings = F.normalize(logic_cl_embeddings, dim=1)
      
      pos_cl_ratings = torch.sum(torch.mul(struc_cl_embeddings, logic_cl_embeddings), dim=-1)
      tot_cl_ratings = torch.matmul(struc_cl_embeddings, torch.transpose(logic_cl_embeddings, 0, 1))

      pos_cl_ratings = torch.exp(pos_cl_ratings / self.tau)
      tot_cl_ratings = torch.sum(torch.exp(tot_cl_ratings / self.tau), dim=1)

      infoNCE_loss = -torch.sum(torch.log(pos_cl_ratings / tot_cl_ratings))
      
      return infoNCE_loss


  def forward(self, latent_vars, task, fast_mode=False, fast_inference_mode=False, pre_logic=False):

    if task == "logic":

      if fast_inference_mode:

        samples = latent_vars
        scores = []

        for ind in range(len(samples)):
          pred_name, pred_sample = samples[ind]

          rel_idx = self.rel2idx[pred_name]

          sample_mat = torch.tensor(pred_sample, dtype=torch.long).to(cmd_args.device) 
          

          head_query = self.node_embedding[sample_mat[:, 0]]
          tail_query = self.node_embedding[sample_mat[:, 1]]

          sample_dist = torch.sum(abs((head_query + self.relation_embedding[rel_idx]) - \
                                      (self.relation_embedding[rel_idx] * tail_query)), 1)
          sample_score = 1. / (torch.exp((sample_dist - self.r) / self.t) + 1.0)
          scores.append(sample_score)
        return scores

      elif fast_mode:

        samples, neg_mask, latent_mask, obs_var, neg_var = latent_vars
        scores = []
        obs_probs = []
        neg_probs = []

        pos_mask_mat = torch.tensor([pred_mask[1] for pred_mask in neg_mask], dtype=torch.float).to(cmd_args.device)
        neg_mask_mat = (pos_mask_mat == 0).type(torch.float)
        latent_mask_mat = torch.tensor([pred_mask[1] for pred_mask in latent_mask], dtype=torch.float).to(cmd_args.device)
        obs_mask_mat = (latent_mask_mat == 0).type(torch.float)

        for ind in range(len(samples)):
          pred_name, pred_sample = samples[ind]
          _, obs_sample = obs_var[ind]
          _, neg_sample = neg_var[ind]

          rel_idx = self.rel2idx[pred_name]

          sample_mat = torch.tensor(pred_sample, dtype=torch.long).to(cmd_args.device)
          obs_mat = torch.tensor(obs_sample, dtype=torch.long).to(cmd_args.device)
          neg_mat = torch.tensor(neg_sample, dtype=torch.long).to(cmd_args.device)

          sample_mat = torch.cat([sample_mat, obs_mat, neg_mat], dim=0)

          head_query = self.node_embedding[sample_mat[:, 0]]
          tail_query = self.node_embedding[sample_mat[:, 1]]
          
          sample_dist = torch.sum(abs((head_query + self.relation_embedding[rel_idx]) - \
                                      (self.relation_embedding[rel_idx] * tail_query)), 1)
          sample_score = 1. / (torch.exp((sample_dist - self.r) / self.t) + 1.0)

          var_prob = sample_score[len(pred_sample):]
          obs_prob = var_prob[:len(obs_sample)]
          neg_prob = var_prob[len(obs_sample):]
          sample_score = sample_score[:len(pred_sample)]

          scores.append(sample_score)
          obs_probs.append(obs_prob)
          neg_probs.append(neg_prob)

        score_mat = torch.stack(scores, dim=0)
        score_mat = score_mat

        pos_score = (1 - score_mat) * pos_mask_mat
        neg_score = score_mat * neg_mask_mat

        potential = 1 - ((pos_score + neg_score) * latent_mask_mat + obs_mask_mat).prod(dim=0)

        obs_mat = torch.cat(obs_probs, dim=0)

        if obs_mat.size(0) == 0:
          obs_loss = 0.0
        else:
          obs_loss = self.xent_loss(obs_mat, torch.ones_like(obs_mat), reduction='sum')

        neg_mat = torch.cat(neg_probs, dim=0)
        if neg_mat.size(0) != 0:
          obs_loss += self.xent_loss(obs_mat, torch.zeros_like(neg_mat), reduction='sum')

        obs_loss /= (obs_mat.size(0) + neg_mat.size(0) + 1e-6)

        return potential, (score_mat * latent_mask_mat).view(-1), obs_loss
    
 
    elif task == "cf":
      
      u_indices, i_indices, final_embedding = latent_vars
      pre_rating = self.get_rating(u_indices, i_indices, final_embedding)
      return pre_rating

    else:

      raise Exception("unknown task mode: " + task)




    

    