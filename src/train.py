import torch
import argparse, os, math, random, warnings
from recommender import *
from dataset import Dataset
from cmd_args import cmd_args
import torch.optim as optim
from graph import KnowledgeGraph
from predicate import PRED_DICT
from utils import get_lr
import numpy as np
from collections import Counter
from rs_data_loader import *
from evaluation import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # ignore the warnings 
warnings.filterwarnings('ignore')

# data for fact
dataset = Dataset(cmd_args.dataset, cmd_args.batchsize, cmd_args.shuffle_sampling)
kg = KnowledgeGraph(dataset.fact_dict, PRED_DICT, dataset, cmd_args)

interaction_data, item_set, edge_index_ex, edge_type_ex, n_relation_ex, edge_index_im, edge_type_im, n_relation_im = load_rs_data(cmd_args)
train_data, eval_data, test_data = interaction_data

rs_model = SLHR(kg, cmd_args.embedding_size, n_relation_ex, n_relation_im).to(cmd_args.device)
optimizer = optim.Adam(rs_model.parameters(), lr=cmd_args.learning_rate, weight_decay=cmd_args.l2_coef)
rs_criterion = nn.BCELoss()


def train(cmd_args):

  print('preparing data for M-step...')
  pred_arg1_set_arg2 = dict()
  pred_arg2_set_arg1 = dict()
  pred_fact_set = dict()
  for pred in dataset.fact_dict_2:
    pred_arg1_set_arg2[pred] = dict()
    pred_arg2_set_arg1[pred] = dict()
    pred_fact_set[pred] = set()
    for _, args in dataset.fact_dict_2[pred]:
      if args[0] not in pred_arg1_set_arg2[pred]:
        pred_arg1_set_arg2[pred][args[0]] = set()
      if args[1] not in pred_arg2_set_arg1[pred]:
        pred_arg2_set_arg1[pred][args[1]] = set()
      pred_arg1_set_arg2[pred][args[0]].add(args[1])
      pred_arg2_set_arg1[pred][args[1]].add(args[0])
      pred_fact_set[pred].add(args)

  grounded_rules = []
  for rule_idx, rule in enumerate(dataset.rule_ls):
    grounded_rules.append(set())
    body_atoms = []
    head_atom = None
    for atom in rule.atom_ls:
      if atom.neg:
        body_atoms.append(atom)
      elif head_atom is None:
        head_atom = atom
    
    assert len(body_atoms) <= 2
    
    if len(body_atoms) > 0:
      body1 = body_atoms[0]
      for _, body1_args in dataset.fact_dict_2[body1.pred_name]:
        var2arg = dict()
        var2arg[body1.var_name_ls[0]] = body1_args[0]
        var2arg[body1.var_name_ls[1]] = body1_args[1]
        for body2 in body_atoms[1:]:
          if body2.var_name_ls[0] in var2arg:
            if var2arg[body2.var_name_ls[0]] in pred_arg1_set_arg2[body2.pred_name]:
              for body2_arg2 in pred_arg1_set_arg2[body2.pred_name][var2arg[body2.var_name_ls[0]]]:
                var2arg[body2.var_name_ls[1]] = body2_arg2
                grounded_rules[rule_idx].add(tuple(sorted(var2arg.items())))
          elif body2.var_name_ls[1] in var2arg:
            if var2arg[body2.var_name_ls[1]] in pred_arg2_set_arg1[body2.pred_name]:
              for body2_arg1 in pred_arg2_set_arg1[body2.pred_name][var2arg[body2.var_name_ls[1]]]:
                var2arg[body2.var_name_ls[0]] = body2_arg1
                grounded_rules[rule_idx].add(tuple(sorted(var2arg.items())))

  print("Collect head atoms derived by grounded formulas")
  grounded_obs = dict()
  grounded_hid = dict()
  grounded_hid_score = dict()
  cnt_hid = 0
  for rule_idx in range(len(dataset.rule_ls)):
    print("rule_idx:", rule_idx)
    rule = dataset.rule_ls[rule_idx]
    for var2arg in grounded_rules[rule_idx]:
      var2arg = dict(var2arg)
      head_atom = rule.atom_ls[-1]
      assert not head_atom.neg    # head atom
      pred = head_atom.pred_name
      args = (var2arg[head_atom.var_name_ls[0]], var2arg[head_atom.var_name_ls[1]])
      if args in pred_fact_set[pred]:
        if (pred, args) not in grounded_obs:
          grounded_obs[(pred, args)] = []
        grounded_obs[(pred, args)].append(rule_idx)
      else:
        if (pred, args) not in grounded_hid:
          grounded_hid[(pred, args)] = []
        grounded_hid[(pred, args)].append(rule_idx)
  print('observed: %d, hidden: %d' % (len(grounded_obs), len(grounded_hid)))


  pred_aggregated_hid = dict()
  pred_aggregated_hid_args = dict()
  for (pred, args) in grounded_hid:
    if pred not in pred_aggregated_hid:
      pred_aggregated_hid[pred] = []
    if pred not in pred_aggregated_hid_args:
      pred_aggregated_hid_args[pred] = []
    pred_aggregated_hid[pred].append((dataset.const2ind[args[0]], dataset.const2ind[args[1]]))
    pred_aggregated_hid_args[pred].append(args)
  pred_aggregated_hid_list = [[pred, pred_aggregated_hid[pred]] for pred in sorted(pred_aggregated_hid.keys())]

  for current_epoch in range(cmd_args.num_epochs):

    print("********** Training: E-step **********")
    num_batches = int(math.ceil(len(dataset.test_fact_ls) / cmd_args.batchsize))

    acc_loss = 0.0
    cur_batch = 0

    for samples_by_r, latent_mask_by_r, neg_mask_by_r, obs_var_by_r, neg_var_by_r in \
        dataset.get_batch_by_q(cmd_args.batchsize):

      loss = 0.0
      r_cnt = 0
      for ind, samples in enumerate(samples_by_r):
        neg_mask = neg_mask_by_r[ind]
        latent_mask = latent_mask_by_r[ind]
        obs_var = obs_var_by_r[ind]
        neg_var = neg_var_by_r[ind]

        if sum([len(e[1]) for e in neg_mask]) == 0:
          continue

        potential, posterior_prob, obs_xent = rs_model([samples, neg_mask, latent_mask, obs_var, neg_var],
                                                        task = "logic", fast_mode=True)

        if cmd_args.no_entropy == 1:
          entropy = 0
        else:
          # print("entropy")
          entropy = compute_entropy(posterior_prob) / cmd_args.entropy_temp

        loss += - (potential.sum() * dataset.rule_ls[ind].weight + entropy) / (potential.size(0) + 1e-6) + obs_xent

        r_cnt += 1

      if r_cnt > 0:
        loss /= r_cnt
        acc_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

      
      cur_batch += 1
    print('Epoch %d, Fact train loss: %.4f, lr: %.4g' % (current_epoch, acc_loss / cur_batch, get_lr(optimizer)))


    print("********** Training: M-step **********")
    with torch.no_grad():

      posterior_prob = rs_model(pred_aggregated_hid_list, task = "logic", fast_inference_mode=True)
      for pred_i, (pred, var_ls) in enumerate(pred_aggregated_hid_list):
        for var_i, var in enumerate(var_ls):
          args = pred_aggregated_hid_args[pred][var_i]
          grounded_hid_score[(pred, args)] = posterior_prob[pred_i][var_i]

      rule_weight_gradient = torch.zeros(len(dataset.rule_ls))
      for (pred, args) in grounded_obs:
        for rule_idx in set(grounded_obs[(pred, args)]):
          rule_weight_gradient[rule_idx] += 1.0 - compute_MB_proba(dataset.rule_ls, grounded_obs[(pred, args)])
      for (pred, args) in grounded_hid:
        for rule_idx in set(grounded_hid[(pred, args)]):
          target = grounded_hid_score[(pred, args)]
          rule_weight_gradient[rule_idx] += target - compute_MB_proba(dataset.rule_ls, grounded_hid[(pred, args)])

      print("Epoch " + str(current_epoch), end=',')
      for rule_idx, rule in enumerate(dataset.rule_ls):
        rule.weight += cmd_args.learning_rate_rule_weights * rule_weight_gradient[rule_idx]
        print(" Rule weight: " + str(dataset.rule_ls[rule_idx].weight), end=' ')
    print()

    print("********** Training: RS task **********")
    cf_data_loader = Data.DataLoader(dataset=train_data, batch_size = cmd_args.cf_batch_size, shuffle = True)
    epoch_loss = 0  
    for i, batch in enumerate(cf_data_loader):
      user, item, rating = batch[:,0].long(), batch[:,1].long(), batch[:,2].float()
      
      final_embeddings = rs_model.get_struc_embedding(edge_index_ex, edge_type_ex, edge_index_im, edge_type_im)
      pre_rating = rs_model([user, item, final_embeddings], task = "cf")
      pre_rating = pre_rating.view(-1)
      # print("pre_rating:", pre_rating)
      loss = rs_criterion(pre_rating, rating)

      cl_list = torch.from_numpy(np.array(random.choices(range(0, kg.num_ents), k=user.size(0))))
      ssl_loss = rs_model.cal_ssl_loss(cl_list, final_embeddings)
      # print("ssl_loss:", ssl_loss)

      loss = loss + cmd_args.lambda_ssl * ssl_loss

      batch_loss = loss.item()
      epoch_loss = epoch_loss + batch_loss

      # Backward and optimize
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    print('Epoch %d, RS train loss: %.4f, lr: %.4g' % (current_epoch, epoch_loss , get_lr(optimizer)))
    test(current_epoch, rs_model)


def test(epoch, rs_model):

    print("********** Testing: RS task **********")

    rs_model.eval()

    final_embeddings = rs_model.get_struc_embedding(edge_index_ex, edge_type_ex, edge_index_im, edge_type_im)
    
    start = 0
    train_auc_list = []
    train_acc_list = []
    train_f1_list  = []

    while start + cmd_args.cf_batch_size <= train_data.shape[0]:
        pre_train = rs_model.get_test_rating(train_data[start:start+cmd_args.cf_batch_size, 0].long(), train_data[start:start+cmd_args.cf_batch_size, 1].long(), final_embeddings, cmd_args.like_name, cmd_args.pre_alpha)
        train_auc, train_acc, train_f1 = auc_acc(train_data[start:start+cmd_args.cf_batch_size, 2].data.numpy(), pre_train.data.numpy())

        train_auc_list.append(train_auc)
        train_acc_list.append(train_acc)
        train_f1_list.append(train_f1)

        start += cmd_args.cf_batch_size

    log_train = 'Train set results:  '+'train_AUC: {:.4f}'.format(float(np.mean(train_auc_list)))+'  train_ACC: {:.4f}'.format(float(np.mean(train_acc_list)))+'  train_F1: {:.4f}'.format(float(np.mean(train_f1_list)))
    print(log_train)

    start = 0
    eval_auc_list = []
    eval_acc_list = []
    eval_f1_list  = []

    while start + cmd_args.cf_batch_size <= eval_data.shape[0]:
        pre_eval = rs_model.get_test_rating(eval_data[start:start+cmd_args.cf_batch_size, 0].long(), eval_data[start:start+cmd_args.cf_batch_size, 1].long(), final_embeddings, cmd_args.like_name, cmd_args.pre_alpha)
        eval_auc, eval_acc, eval_f1 = auc_acc(eval_data[start:start+cmd_args.cf_batch_size, 2].data.numpy(), pre_eval.data.numpy())

        eval_auc_list.append(eval_auc)
        eval_acc_list.append(eval_acc)
        eval_f1_list.append(eval_f1)

        start += cmd_args.cf_batch_size

    log_eval = 'Eval set results:  '+'eval_AUC: {:.4f}'.format(float(np.mean(eval_auc_list)))+'  eval_ACC: {:.4f}'.format(float(np.mean(eval_acc_list)))+'  eval_F1: {:.4f}'.format(float(np.mean(eval_f1_list)))
    print(log_eval)
    

    start = 0
    test_auc_list = []
    test_acc_list = []
    test_f1_list  = []

    while start + cmd_args.cf_batch_size <= test_data.shape[0]:
        pre_test = rs_model.get_test_rating(test_data[start:start+cmd_args.cf_batch_size, 0].long(), test_data[start:start+cmd_args.cf_batch_size, 1].long(), final_embeddings, cmd_args.like_name, cmd_args.pre_alpha)
        test_auc, test_acc, test_f1 = auc_acc(test_data[start:start+cmd_args.cf_batch_size, 2].data.numpy(), pre_test.data.numpy())

        test_auc_list.append(test_auc)
        test_acc_list.append(test_acc)
        test_f1_list.append(test_f1)

        start += cmd_args.cf_batch_size

    log_test = 'Test set results:  '+'test_AUC: {:.4f}'.format(float(np.mean(test_auc_list)))+'  test_ACC: {:.4f}'.format(float(np.mean(test_acc_list)))+'  test_F1: {:.4f}'.format(float(np.mean(test_f1_list)))
    print(log_test)
    
    
    f = open(log_path, 'a')
    f.write("Epoch:" + str(epoch) + '\n')
    f.write(log_train + '\n')
    f.write(log_eval + '\n')
    f.write(log_test + '\n')
    f.close()

    



def compute_entropy(posterior_prob):
  eps = 1e-6
  posterior_prob.clamp_(eps, 1 - eps)
  compl_prob = 1 - posterior_prob
  entropy = -(posterior_prob * torch.log(posterior_prob) + compl_prob * torch.log(compl_prob)).sum()
  return entropy


def compute_MB_proba(rule_ls, ls_rule_idx):
  rule_idx_cnt = Counter(ls_rule_idx)
  numerator = 0
  for rule_idx in rule_idx_cnt:
    weight = rule_ls[rule_idx].weight
    cnt = rule_idx_cnt[rule_idx]
    try:
    	if weight * cnt > 0:
    		if weight * cnt > 0.02:
    			numerator += 1.02
    		else:
    			numerator += math.exp(weight * cnt)
    	else:
    		if weight * cnt < -10:
    			numerator += 0.00004
    		else:
    			numerator += math.exp(weight * cnt)
    except OverflowError:
    	if weight * cnt > 0:
    		numerator += 1.02
    	else:
    		numerator += 0.00004
    		
  return numerator / (numerator + 1.0)

# def compute_MB_proba(rule_ls, ls_rule_idx):
#   rule_idx_cnt = Counter(ls_rule_idx)
#   numerator = 0
#   for rule_idx in rule_idx_cnt:
#     weight = rule_ls[rule_idx].weight
#     cnt = rule_idx_cnt[rule_idx]
#     numerator += math.exp(weight * cnt)
#   return numerator / (numerator + 1.0)


if __name__ == '__main__':
  random.seed(cmd_args.seed)
  np.random.seed(cmd_args.seed)
  torch.manual_seed(cmd_args.seed)

  log_path = 'log/' + cmd_args.dataset + '-' + cmd_args.GNN_type + '.txt'
  f = open(log_path, 'a') 
  f.write('\n\n' + str(cmd_args) + '\n')
  f.close()

  train(cmd_args)
