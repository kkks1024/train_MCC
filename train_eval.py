import glob
import torch
import math
import time
import logging

from torch import gather, nn
from torch import optim
from torch.optim import Adam
import numpy as np
import torch.autograd as autograd

from data_gen import *
from data import *
from perceiver.encoder import PerceiverEncoder
from perceiver.decoder import PerceiverDecoder
from perceiver.perceiver import PerceiverIO
from perceiver.query import Query_Gen
from perceiver.query_new2 import Query_Gen_transformer, Query_Gen_transformer_PE
from util.epoch_timer import epoch_time
from util.look_table import lookup_value_1d, lookup_value_close, lookup_value_average, lookup_value_bilinear, lookup_value_grid

from perceiver.encoder import PerceiverEncoder
from perceiver.decoder import PerceiverDecoder
from perceiver.perceiver import PerceiverIO
from tensorboardX import SummaryWriter
import os

latent_dim = 256
latent_num = 256
input_dim = 1
batchsize = 64  #### num of distributions trained in one epoch
seq_len = 1204  #### number of data points sampled from each distribution
decoder_query_dim = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

learning_rate = 0.0002


encoder = PerceiverEncoder(
    input_dim=input_dim,
    latent_num=latent_num,
    latent_dim=latent_dim,
    cross_attn_heads=8,
    self_attn_heads=16,
    num_self_attn_per_block=8,
    num_self_attn_blocks=1
)

decoder = PerceiverDecoder(
    q_dim=decoder_query_dim,
    latent_dim=latent_dim,
)

query_gen = Query_Gen_transformer(
    input_dim = input_dim,
    dim = decoder_query_dim
)

model1 = PerceiverIO(encoder=encoder, decoder=decoder, query_gen = query_gen, decoder_query_dim = decoder_query_dim).to(device)
model2 = PerceiverIO(encoder=encoder, decoder=decoder, query_gen = query_gen, decoder_query_dim = decoder_query_dim).to(device)
model1.load_state_dict(torch.load('saved/MOG1/model1_data_order_milb_bs30-GMM-170000--0.18.pt', map_location=device))
model2.load_state_dict(torch.load('saved/MOG1/model2_data_order_milb_bs30-GMM-170000--0.18.pt', map_location=device))

logging.basicConfig(filename='logs/train.log', level=logging.INFO)
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform(m.weight.data)

def ma(a, window_size=100):
    return [np.mean(a[i:i+window_size]) for i in range(0,len(a)-window_size)]

count_params = count_parameters(model1) + count_parameters(model2)
print(f'The model has {count_params:,} trainable parameters')
#model.apply(initialize_weights)

params = [{'params': (list(model1.parameters()) + list(model2.parameters()))}]
optimizer = optim.Adam(params, lr=learning_rate)

class logWriter(object):
    def __init__(self, logdir="save/logs"):
       
        super().__init__()
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        self.writer = SummaryWriter(logdir)

    def record(self, tag, loss_item: dict, step: int):
        
        #for key,value in loss_item:
            #self.writer.add_scalar(tag=key, scalar_value=value,global_step=step)
        self.writer.add_scalar(tag=tag, scalar_value=loss_item, global_step=step)

def mutual_information(joint, model1, model2):


    joint_x =(joint[:,:,0]).unsqueeze(2)
    joint_y =(joint[:,:,1]).unsqueeze(2)
    #joint_x = (joint_x.unsqueeze(0)).unsqueeze(2)
    #joint_y = (joint_y.unsqueeze(0)).unsqueeze(2)
    #print(joint_x.shape)
    matrix_x = model1(joint_x)
    matrix_y = model2(joint_y)

    f = lookup_value_1d(matrix_x, joint_x)
    g = lookup_value_1d(matrix_y, joint_y)

    layer_norm = nn.LayerNorm(seq_len, eps=1e-5, device=device)
    f = layer_norm(f)
    g = layer_norm(g)
    #print(matrix_y)

    #print('===================> matrix range is:', torch.max(j_matrix), torch.min(j_matrix))
    value = f * g
    value = torch.mean(value,dim=1)
  

    #print(value)
    return value

def infer(model1, model2, batch):
    model1.eval()
    model2.eval()
    batch = torch.tensor(batch, dtype=torch.float32, device=device)
    with torch.no_grad():
        e_cor = mutual_information(batch, model1, model2)
        e_cor = torch.mean(e_cor)

    return e_cor


def compare(MI_XY, MI_XZ):
    comp = []
    for i in range(len(MI_XY)):
        if MI_XY[i] >= MI_XZ[i]:
            comp.append(1)
        else:
            comp.append(0)
    return comp


def save_to_file(i, acc):
    with open("eval_result/test_bs30.txt", "a") as f:
        output_string = f"3,bs=30 order test acc for model_{i} is {acc}\n"
        f.write(output_string)
        print(output_string)


def evaluate(model1, model2, path, i):
    # npy_datas = [f for f in os.listdir(os.path.join(path, "data/mix_gauss")) if f.endswith('.npy')]
    # npy_datas = [f for f in os.listdir(os.path.join(path, "data")) if f.endswith('.npy')]

    results_xy = []
    results_xz = []
    MI_XY = []
    MI_XZ = []

    k = 0
    with open(os.path.join(path, "weight.pkl"), 'rb') as f:
        weights = pickle.load(f)

    with open(os.path.join(path, "mean.pkl"), 'rb') as f:
        means = pickle.load(f)

    with open(os.path.join(path, "cov.pkl"), 'rb') as f:
        covs = pickle.load(f)
    times = []    
    for idx in range(2000):
        k += 1
        #if k > 10:
            #break
            # 提取加载的参数
        gm = GaussianMixture(n_components=5)
        gm.weights_ = np.array(weights[idx])
        gm.means_ = np.array(means[idx])
        gm.covariances_ = np.array(covs[idx])
        infer_xy = []
        infer_xz = []
        data_xy = np.zeros((batchsize, seq_len,  2))
        data_xz = np.zeros((batchsize, seq_len,  2))
        for j in range(batchsize):
            # gm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(covs)).transpose((0, 2, 1))

            joint_samples, labels = gm.sample(n_samples=seq_len)

            joint_samples = np.array(joint_samples)
            joint_xy_samples = joint_samples[:, [0, 1]]
            joint_xz_samples = joint_samples[:, [0, 2]]

            '''
            marginal_samples, labels = gm.sample(n_samples=seq_len)
            y_marginal = marginal_samples[:, 1]
            z_marginal = marginal_samples[:, 2]
            marginal_xy_samples = np.column_stack((joint_samples[:, 0], y_marginal))
            marginal_xz_samples = np.column_stack((joint_samples[:, 0], z_marginal))
            '''

            #sample_xy = np.concatenate((joint_xy_samples, marginal_xy_samples), axis=1)
            #sample_xz = np.concatenate((joint_xz_samples, marginal_xz_samples), axis=1)
            sample_xy = joint_xy_samples
            sample_xz = joint_xz_samples
            sample_xy = scale_data(sample_xy)
            sample_xz = scale_data(sample_xz)

            data_xy[j] = np.array(sample_xy)
            data_xz[j] = np.array(sample_xz)
        start_time = time.time()
        infer_1 = infer(model1, model2, data_xy)
        infer_2 = infer(model1, model2, data_xz)
        end_time = time.time()
        times.append(end_time-start_time)
        #if len(times)%10==0:
        print(np.mean(times))
        #infer_xy.append(infer_1)
        #infer_xz.append(infer_2)
        #print(infer_xy,infer_xz)
        results_xy.append(infer_1)
        results_xz.append(infer_2)


        #if k <= 10:
        print("data ", k, "estimate xy mi is: ", infer_1, "estimate xz mi is: ", infer_2)

    with open(os.path.join(path, "XY_mi_gauss.txt"), 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = float(line.strip())
            MI_XY.append(value)

    with open(os.path.join(path, "XZ_mi_gauss.txt"), 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = float(line.strip())
            MI_XZ.append(value)

    #print(MI_XY[0:10], MI_XZ[0:10])

    comp = compare(MI_XY, MI_XZ)
    comp_c = compare(results_xy, results_xz)
    accuracy_c = 0
    print(len(comp), len(comp_c))
    for j in range(len(comp_c)):
        if comp[j] == comp_c[j]:
            accuracy_c += 1
    accuracy_rate = accuracy_c / len(comp_c) * 100
    print(accuracy_rate)
    save_to_file(i, accuracy_rate)


def train(model1, model2, batch, optimizer, writer, clip=1.0, ma_rate=0.01,iter_num=1, log_freq=10):
    model1.train()
    model2.train()
    epoch_loss = 0
    ma_et = 1
    losses = []
    global global_step
    for i in range(iter_num):
        
        global_step += 1

        batch = torch.tensor(batch, dtype=torch.float32, device=device)

        e_cor = mutual_information(batch, model1, model2)
    
        # unbiasing use moving average
        #print(torch.mean(t), torch.mean(et), (1/ma_et.mean()).detach()*torch.mean(et))
        #loss = -(torch.mean(t) - (1/ma_et.mean()).detach()*torch.mean(et))
        # use biased estimator
        loss = - torch.mean(e_cor)
        
        if (global_step)%(log_freq)==0:            
            writer.add_scalar(tag="loss", scalar_value=loss.item(), global_step=global_step)

        optimizer.zero_grad()
        autograd.backward(loss)
        parameters = list(model1.parameters()) + list(model2.parameters())
        torch.nn.utils.clip_grad_norm_(parameters, clip)
        optimizer.step()

        epoch_loss += loss.item()

        losses.append(loss.item())
        if len(losses) > 100:
            losses = losses[1:]  

        average_loss = sum(losses) / len(losses)
        if (i+1)%10==0:
            print('step :', i, '% , average loss :', average_loss)
        #print('step :', i, '% , loss :', loss.item())
        
    return epoch_loss/iter_num

def run(total_epoch, batch, writer):
    train_losses= []
    for step in range(total_epoch):
        start_time = time.time()

        train_loss = train(model1, model2, batch, optimizer, writer)

        end_time = time.time()

        train_losses.append(train_loss)
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        # if step==(total_epoch-1):
        # torch.save(model.state_dict(), 'saved/model-normal-{0}.pt'.format(train_loss))

        print(f'Epoch: {step + 1} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} ')

    return train_loss

def scale_tensor(input_tensor):
    min_val = np.min(input_tensor)
    max_val = np.max(input_tensor)
    scaled_tensor = 2 * (input_tensor - min_val) / (max_val - min_val) - 1
    return scaled_tensor

if __name__ == '__main__':
    ma_rate = 1.0
    global_step = 0
    #writer = logWriter('logs')
    writer = SummaryWriter('logs/experiment_2')
    #data = np.random.uniform(-1, 1, size=(10000, 2))

    for i in range(1000000):
        batch = gen_batch_mog(batchsize=batchsize, seq_len=seq_len, dim=1, stddev=1)
        # dataset = sample_mixture_gaussians(1000, 500, 1, num_components=10, stddev=1)
        path = "/home/skang/perceiver-test-Gauss/evaluation/test"
        if (i + 1) % 5000 == 0:
            evaluate(model1, model2, path, i+1+170000)

        train_loss = run(total_epoch=1, batch=batch, writer=writer)
        if (i + 1) % 10000 == 0:
            torch.save(model1.state_dict(), f'/home/skang/perceiver-test-Gauss/saved/MOG1/model1_data_order_milb_bs30-GMM-{i+1+170000}-{train_loss:.2f}.pt')
            torch.save(model2.state_dict(), f'/home/skang/perceiver-test-Gauss/saved/MOG1/model2_data_order_milb_bs30-GMM-{i+1+170000}-{train_loss:.2f}.pt')
