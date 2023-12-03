import argparse

cmd_opt = argparse.ArgumentParser(description='argparser')

cmd_opt.add_argument('--dataset', default='test', help='root of data_process')
cmd_opt.add_argument('--seed', default=2023, type=int, help='random seed')
cmd_opt.add_argument('--num_epochs', default=200, type=int, help='num epochs')
cmd_opt.add_argument('--embedding_size', default=16, type=int, help='embedding size')
cmd_opt.add_argument('--learning_rate', default=0.001, type=float, help='learning rate')
cmd_opt.add_argument('--l2_coef', default=2e-5, type=float, help='L2 coefficient for weight decay')
cmd_opt.add_argument('--pre_alpha', default=0.8, type=float, help='for test')
cmd_opt.add_argument('--tau', type=float, default=1.0, help='SSL task.')
cmd_opt.add_argument('--lambda_ssl', type=float, default=0.0001, help='SSL task.')
cmd_opt.add_argument('--show_topk', type=bool, default=True, help='top-k evaluation.')
cmd_opt.add_argument('--device', default='cpu', type=str, help='run on cpu or cuda')

# Fact task
cmd_opt.add_argument('--batchsize', default=256, type=int, help='batch size for training')
cmd_opt.add_argument('--shuffle_sampling', default=1, type=int, help='set 1 to shuffle formula when sampling')
cmd_opt.add_argument('--entropy_temp', default=1, type=float, help='temperature for entropy term')
cmd_opt.add_argument('--no_entropy', default=0, type=int, help='no entropy term in ELBO')
cmd_opt.add_argument('--learning_rate_rule_weights', default=0.01, type=float, help='learning rate of rule weights')
cmd_opt.add_argument('--r', type=float, default=2.0, help='fermi-dirac decoder parameter for lp.')
cmd_opt.add_argument('--t', type=float, default=1.0, help='fermi-dirac decoder parameter for lp.')
cmd_opt.add_argument('--like_name', default="like_name", type=str, help='like_name')

# CF task
cmd_opt.add_argument('--cf_batch_size', default=1024, type=int, help='batch size for cf training')
cmd_opt.add_argument('--GNN_type', default="RGCN", type=str, help='GNN_type')
cmd_opt.add_argument('--trans_type', default="GATEITU", type=str, help='trans_type')
cmd_opt.add_argument('--agg_layer_type', default="mlp", type=str, help='agg_type')


cmd_args, _ = cmd_opt.parse_known_args()

