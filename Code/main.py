import torch
import time
import sys
import utils
from keras.utils.np_utils import *
from sklearn.model_selection import  train_test_split
from  sklearn.model_selection  import  KFold
from model import GTN
import pickle
import argparse
from utils import Accuracy,Aiming,Coverage,Abs_True_Rate,Abs_False_Rate
from tqdm import tqdm
torch.set_printoptions(threshold=np.inf)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str,
                        help='Dataset')
    parser.add_argument('--epoch', type=int, default=250,
                        help='Training Epochs')
    parser.add_argument('--node_dim', type=int, default=150,
                        help='Node dimension')
    parser.add_argument('--num_channels', type=int, default=2,
                        help='number of channels')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001,
                        help='l2 reg')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layer')
    parser.add_argument('--norm', type=str, default='true',
                        help='normalization')
    parser.add_argument('--adaptive_lr', type=str, default='true',
                        help='adaptive learning rate')
    print(time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()))
    args = parser.parse_args()
    print(args)
    epochs = args.epoch
    node_dim = args.node_dim
    num_channels = args.num_channels
    lr = args.lr
    weight_decay = args.weight_decay
    num_layers = args.num_layers
    norm = args.norm
    adaptive_lr = args.adaptive_lr
    # star = time.time()
    with open('data/D_T_D_node_feature.pkl', 'rb') as f:
        node_features = pickle.load(f)
    with open('data/D_T_D_edges.pkl', 'rb') as f:
        edges = pickle.load(f)
    with open('data/D_T_D_labels.pkl', 'rb') as f:
        labels = pickle.load(f)
    num_nodes = edges[0].shape[0]
    for i,edge in enumerate(edges):
        if i ==0:
            A = torch.from_numpy(edge.todense()).type(torch.FloatTensor).unsqueeze(-1)
        else:
            A = torch.cat([A,torch.from_numpy(edge.todense()).type(torch.FloatTensor).unsqueeze(-1)], dim=-1)
    A = torch.cat([A,torch.eye(num_nodes).type(torch.FloatTensor).unsqueeze(-1)], dim=-1)
    node_features = torch.from_numpy(node_features).type(torch.FloatTensor)
    num_classes = 14
    node = labels[0:, 0]
    target = labels[0:, 1:]
    kf = KFold(n_splits=10,shuffle=True,random_state=4)
    n_count = 0
    Acc = []
    Aim = []
    Cov = []
    Abs_T = []
    Abs_F = []
    for train_index, test_index in kf.split(node):
        best_valid_acc = -1
        best_valid_Abs_True_Rate = -1
        best_train_acc = -1
        best_train_Abs_True_Rate = -1
        n_count = n_count + 1
        print('cross validation: ', n_count)
        train_train_index, train_valid_index = train_test_split(train_index, test_size=0.1)
        node_train_train, node_train_valid = node[train_train_index], node[train_valid_index]
        target_train_train, target_trian_valid = target[train_train_index], target[train_valid_index]
        train_train_node = torch.from_numpy(node_train_train).type(torch.LongTensor)
        train_train_target = torch.from_numpy(target_train_train).type(torch.LongTensor)
        train_valid_node = torch.from_numpy(node_train_valid).type(torch.LongTensor)
        train_valid_target = torch.from_numpy(target_trian_valid).type(torch.LongTensor)
        node_test =node[test_index]
        target_test = target[test_index]
        test_node = torch.from_numpy(node_test).type(torch.LongTensor)
        test_target = torch.from_numpy(target_test).type(torch.LongTensor)
        # for l in range(1):
        #     model = GTN(num_edge=A.shape[-1],
        #                 num_channels=num_channels,
        #                 w_in=node_features.shape[1],
        #                 w_out=node_dim,
        #                 num_class=num_classes,
        #                 num_layers=num_layers,
        #                 norm=norm)
        #     if adaptive_lr == 'false':
        #         optimizer = torch.optim.Adam(model.parameters(), lr=0.005, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001, amsgrad=False)
        #     else:
        #         optimizer = torch.optim.Adam([{'params': model.weight},
        #                                       {'params': model.linear1.parameters()},
        #                                       {'params': model.linear2.parameters()},
        #                                       {"params": model.layers.parameters(), "lr": 0.5}
        #                                       ], lr=0.005, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001,
        #                                      amsgrad=False)
        #     for i in tqdm(range(epochs)):
        #         for param_group in optimizer.param_groups:
        #             if param_group['lr'] > 0.005:
        #                 param_group['lr'] = param_group['lr'] * 0.9
        #         # print('Epoch:  ', i + 1)
        #         star = time.time()
        #         model.zero_grad()
        #         model.train()
        #         loss, y_train, Ws = model(A, node_features, train_train_node, train_train_target)
        #         zero = torch.zeros_like(y_train)
        #         one = torch.ones_like(y_train)
        #         train_prediction = torch.where(y_train < 0, zero, one)
        #         train_acc = Accuracy(train_prediction, train_train_target, len(train_train_node))
        #         train_Aiming = Aiming(train_prediction, train_train_target, len(train_train_node))
        #         train_Coverage = Coverage(train_prediction, train_train_target, len(train_train_node))
        #         train_Abs_True_Rate = Abs_True_Rate(train_prediction, train_train_target, len(train_train_node))
        #         train_Abs_False_Rate = Abs_False_Rate(train_prediction, train_train_target, len(train_train_node))
        #         time_cost = time.time() - star
        #         # print('train_acc={:.4},train_Aiming={:.4},train_Coverage={:.4},train_Abs_True_Rate={:.4},train_Abs_False_Rate={:.4},Time={:.4}'.format(train_acc, train_Aiming, train_Coverage, train_Abs_True_Rate, train_Abs_False_Rate,time_cost))
        #         star = time.time()
        #         loss.backward(loss.clone().detach())
        #         optimizer.step()
        #         model.eval()
        #         # Vaild
        #         with torch.no_grad():
        #             star1 = time.time()
        #             valid_loss, y_valid, W = model.forward(A, node_features, train_valid_node, train_valid_target)
        #             zero = torch.zeros_like(y_valid)
        #             one = torch.ones_like(y_valid)
        #             valid_prediction = torch.where(y_valid < 0, zero, one)
        #             valid_acc = Accuracy(valid_prediction, train_valid_target, len(train_valid_node))
        #             valid_Aiming = Aiming(valid_prediction, train_valid_target, len(train_valid_node))
        #             valid_Coverage = Coverage(valid_prediction, train_valid_target, len(train_valid_node))
        #             valid_Abs_True_Rate = Abs_True_Rate(valid_prediction, train_valid_target, len(train_valid_node))
        #             valid_Abs_False_Rate = Abs_False_Rate(valid_prediction, train_valid_target, len(train_valid_node))
        #             time_cost1 = time.time() - star1
        #             # print('valid_acc={:.4},valid_Aiming={:.4},valid_Coverage={:.4},valid_Abs_True_Rate={:.4},valid_Abs_False_Rate={:.4},Time={:.4}'.format(valid_acc, valid_Aiming, valid_Coverage, valid_Abs_True_Rate, valid_Abs_False_Rate,time_cost1))
        #             star1 = time.time()
        #         if valid_acc >= best_valid_acc :
        #             best_valid_acc = valid_acc
        #             best_valid_Aiming = valid_Aiming
        #             best_valid_Coverage = valid_Coverage
        #             best_valid_Abs_False_Rate = valid_Abs_False_Rate
        #             best_valid_Abs_True_Rate = valid_Abs_True_Rate
        #             path = './GTN_model_save-' + str(n_count)
        #             torch.save(model.state_dict(), path)
        #     print('---------------Best Results--------------------')
        #     print(
        #         'Best_valid_acc={:.4},Best_valid_Aiming={:.4},Best_valid_Coverage={:.4},Best_valid_Abs_True_Rate={:.4},Best_valid_Abs_False_Rate={:.4}'.format(
        #             best_valid_acc, best_valid_Aiming, best_valid_Coverage, best_valid_Abs_True_Rate,
        #             best_valid_Abs_False_Rate))
            # Test
        path = 'save_model/GTN_model_save-' + str(n_count)
        model = GTN(num_edge=A.shape[-1], num_channels=num_channels, w_in=node_features.shape[1], w_out=node_dim, num_class=num_classes, num_layers=num_layers, norm=norm)
        model.load_state_dict(torch.load(path))
        model.eval()
        with torch.no_grad():
            star2 = time.time()
            test_loss, y_test, W = model.forward(A, node_features, test_node, test_target)
            zero = torch.zeros_like(y_test)
            one = torch.ones_like(y_test)
            test_prediction = torch.where(y_test < 0, zero, one)
            test_acc = Accuracy(test_prediction, test_target, len(test_node))
            test_Aiming = Aiming(test_prediction, test_target, len(test_node))
            test_Coverage = Coverage(test_prediction, test_target, len(test_node))
            test_Abs_True_Rate = Abs_True_Rate(test_prediction, test_target, len(test_node))
            test_Abs_False_Rate = Abs_False_Rate(test_prediction, test_target, len(test_node))
            time_cost2 = time.time() - star2
            print(
                'test_acc={:.4},test_Aiming={:.4},test_Coverage={:.4},test_Abs_True_Rate={:.4},test_Abs_False_Rate={:.4},Time={:.4}'.format(
                    test_acc, test_Aiming, test_Coverage, test_Abs_True_Rate, test_Abs_False_Rate,time_cost2))
            star2 = time.time()
        Acc.append(test_acc)
        Aim.append(test_Aiming)
        Cov.append(test_Coverage)
        Abs_T.append(test_Abs_True_Rate)
        Abs_F.append(test_Abs_False_Rate)
    acc_avg = np.average(Acc)
    Aim_avg = np.average(Aim)
    Cov_avg = np.average(Cov)
    Abs_T_avg = np.average(Abs_T)
    Abs_F_avg = np.average(Abs_F)

    print('************************************************************')
    print('Accuracy:{:.4},Aiming:{:.4},Coverage:{:.4},Abs_True_Rate:{:.4},Abs_False_Rate:{:.4}'.format(acc_avg,Aim_avg,Cov_avg,Abs_T_avg,Abs_F_avg))
    print(time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()))
