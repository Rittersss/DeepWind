import numpy as np
import pandas as pd
import os
import pickle
import torch.nn.functional as F

from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

import random
import torch
import torch.nn as nn
import torch.optim as optim
import math
from torch.utils.data import DataLoader, Dataset
import torch.optim.lr_scheduler as lr_scheduler

from sklearn.model_selection import GroupKFold

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic=True


#save object
def save_obj(obj, name):
    with open(name + '.pickle', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

# load object
def load_obj(name ):
    with open(name + '.pickle', 'rb') as f:
        return pickle.load(f)

# convert wind direction angle
def transform_wdir(sin, cos):
    wdir = np.arctan2(sin, cos)/2/np.pi*360
    wdir = np.where(wdir > 360, wdir - 360, wdir)
    wdir = np.where(wdir < 0, wdir + 360, wdir)
    return wdir

# set hour labels
def TagHour(sample_id):
    sample_id = int(sample_id[6:])
    shift =  (sample_id+1) % 2 
    if shift == 0:
        hour_tag = list(range(0,24))*2
    if shift == 1:
        hour_tag = list(range(12,24)) + list(range(0,24)) + list(range(0,12))
    assert len(hour_tag) == 48
    return hour_tag

# feature processing
def gen_feats(data, train=True):
    data = pd.concat([data, data['ID'].str.split('_', expand=True).rename(columns={0:'station',1:'sample',2:'time'})], axis=1)
    
    drop_feats = [
        'skt', 'sst',
        't_L500', 't_L700',  'deg0l', 't_L950', 't_L925', 't_L1000', '2t', 't_L850', 't_L900', '2d',
        # 'q_L1000', 'q_L200', 'r_L200', 'L200_dir',
    ]
    data.drop(drop_feats, axis=1, inplace=True)
    
    log_feats = [
        'cape',
        'capes',
        'cp',
    ]
    data[log_feats] = np.log1p(data[log_feats])

    cat_feats = []

    # Data augmentation
    data['station'] = data['station'].apply(lambda x: int(x.split('D')[-1])) - 1
    data['sample'] = data['sample'].apply(lambda x: int(x.split('Sample')[-1]))
    data['time'] = data['time'].astype(int) - 1
    
    data['hour'] = ((list(range(12,24)) + list(range(0,24)) + list(range(0,12)))*14+(list(range(0,24))*2)*14)*(1178//2)
    cat_feats += ['station', 'time', 'hour'] #['station']
    
    data['real_sample'] = data['sample'] + data['hour']
    

    # Transformation of wind UV components
    data['10_spd'] = np.sqrt(data['10u']**2+data['10v']**2)
    data['100_spd'] = np.sqrt(data['100u']**2+data['100v']**2)
    data['L200_spd'] = np.sqrt(data['u_L200']**2+data['v_L200']**2)
    data['L500_spd'] = np.sqrt(data['u_L500']**2+data['v_L500']**2)
    data['L700_spd'] = np.sqrt(data['u_L700']**2+data['v_L700']**2)
    data['L850_spd'] = np.sqrt(data['u_L850']**2+data['v_L850']**2)
    data['L900_spd'] = np.sqrt(data['u_L900']**2+data['v_L900']**2)
    data['L925_spd'] = np.sqrt(data['u_L925']**2+data['v_L925']**2)
    data['L950_spd'] = np.sqrt(data['u_L950']**2+data['v_L950']**2)
    data['L1000_spd'] = np.sqrt(data['u_L1000']**2+data['v_L1000']**2)


    data['10_dir'] = 180.0 + np.arctan2(data['10u'], data['10v'])*180.0/np.pi
    data['100_dir'] = 180.0 + np.arctan2(data['100u'], data['100v'])*180.0/np.pi
    data['L200_dir'] = 180.0 + np.arctan2(data['u_L200'], data['v_L200'])*180.0/np.pi
    data['L500_dir'] = 180.0 + np.arctan2(data['u_L500'], data['v_L500'])*180.0/np.pi
    data['L700_dir'] = 180.0 + np.arctan2(data['u_L700'], data['v_L700'])*180.0/np.pi
    data['L850_dir'] = 180.0 + np.arctan2(data['u_L850'], data['v_L850'])*180.0/np.pi
    data['L900_dir'] = 180.0 + np.arctan2(data['u_L900'], data['v_L900'])*180.0/np.pi
    data['L925_dir'] = 180.0 + np.arctan2(data['u_L925'], data['v_L925'])*180.0/np.pi
    data['L950_dir'] = 180.0 + np.arctan2(data['u_L950'], data['v_L950'])*180.0/np.pi
    data['L1000_dir'] = 180.0 + np.arctan2(data['u_L1000'], data['v_L1000'])*180.0/np.pi

    if train:
        data['wdir_2min'] = data['wdir_2min'] - data['10_dir']
        data['wdir_2min'] = np.where(data['wdir_2min'] > 180, data['wdir_2min'] - 360, data['wdir_2min'])
        data['wdir_2min'] = np.where(data['wdir_2min'] < -180, data['wdir_2min'] + 360, data['wdir_2min'])

    return data, cat_feats

# Dataset
class MyDataset(Dataset):
    def __init__(self, con_data, cat_data, labels):
        self.con_data = con_data
        self.cat_data = cat_data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.con_data[idx], self.cat_data[idx], self.labels[idx]
   
# CNN Model
class CNNModel(nn.Module):
    def __init__(self, num_con_features):
        super(CNNModel, self).__init__()
        
        # embedding for category variables(station,time,hour)
        self.embedding_layer_station = nn.Embedding(14, 2) #862
        self.embedding_layer_step = nn.Embedding(48, 2)
        self.embedding_layer_hour = nn.Embedding(24,2)
        
        self.lr_embeding = nn.Linear(num_con_features+2+2+2, 256) # 加embedding层的维数，映射为256

        # 1D-CNN
        self.before_layer = nn.Sequential(
            nn.Conv1d(4, 8, 3, padding=1, stride=2),
            nn.ELU(),
            nn.Conv1d(8, 8, 3, padding=1, stride=2),
            nn.ELU(),
            # nn.BatchNorm1d(32),
            nn.Conv1d(8, 8, 3, padding=1),
            nn.ELU(),
            # nn.BatchNorm1d(32),
            nn.Conv1d(8, 8, 3, padding=1),
            nn.ELU(),
            # nn.BatchNorm1d(32),
            nn.Conv1d(8, 4, 3, padding=1),
            nn.ELU(),
        )

        # 1D-CNN
        self.con_layer = nn.Sequential(
            nn.Conv1d(64, 32, 3, padding=1), #64
            nn.ELU(),
            # nn.BatchNorm1d(32),
            nn.Conv1d(32, 32, 3, padding=1),
            nn.ELU(),
            # nn.BatchNorm1d(32),
            nn.Conv1d(32, 32, 3, padding=1),
            nn.ELU(),
            # nn.BatchNorm1d(32),
            nn.Conv1d(32, 32, 3, padding=1),
            nn.ELU(),
            # nn.BatchNorm1d(32),
            nn.Conv1d(32, 32, 3, padding=1),
            nn.ELU(),
            # nn.BatchNorm1d(32),
        )

        # Linear
        self.final_layer = nn.Sequential(
            nn.Linear(32, 32),
            nn.ELU(),
            # nn.BatchNorm1d(32),
            nn.Linear(32, 32),
            nn.ELU(),
            # nn.BatchNorm1d(32),
            nn.Linear(32, 4),
        )

    def forward(self, con_input, cat_input):
        
        # embedding for category variables(station,time,hour)
        station_output = self.embedding_layer_station(cat_input[..., 0].long())
        step_output = self.embedding_layer_step(cat_input[..., 1].long())
        hour_output = self.embedding_layer_hour(cat_input[..., 2].long())
        
        cat_output = torch.cat([station_output, step_output, hour_output], dim=-1)
        output = torch.cat([con_input, cat_output], dim=-1)
        con_output = self.lr_embeding(output) #256
        con_output = torch.reshape(con_output, (-1,64,4))
        
        # CNN operation
        con_output = self.before_layer(con_output.permute(0,2,1)).permute(0,2,1)
        con_output = torch.reshape(con_output, (-1,48,64)) #12, 16*4
        output = self.con_layer(con_output.permute(0,2,1)).permute(0,2,1)
        output = self.final_layer(output)

        return output, cat_output

# GRU Model
class GRUModel(nn.Module):
    def __init__(self, feature_size, hidden_size, num_layers, output_size):
        super(GRUModel, self).__init__()
 
        self.hidden_size = hidden_size  # hidden layer size
        self.num_layers = num_layers  # number of GRU layers

        # feature_size represents the feature dimension, which is the number of features corresponding to each time point, and here it is 72.
        self.gru = nn.GRU(
            feature_size, 
            hidden_size, 
            num_layers, 
            batch_first=True
        )

        # embedding for category variables(station,time,hour)
        self.embedding_layer_station = nn.Embedding(14, 2)
        self.embedding_layer_step = nn.Embedding(48, 2)
        self.embedding_layer_hour= nn.Embedding(24, 2)

        self.mlp = nn.Sequential(
              nn.Linear(hidden_size, output_size),
              nn.Sigmoid()
        )

    def forward(self, con_input, cat_input, hidden=None):
        batch_size = con_input.shape[0]
        
        # embedding for category variables(station,time,hour)
        station_output = self.embedding_layer_station(cat_input[..., 0].long())
        step_output = self.embedding_layer_step(cat_input[..., 1].long())
        hour_output = self.embedding_layer_hour(cat_input[..., 2].long())
        
        cat_output = torch.cat([station_output, step_output, hour_output], dim=-1)
        x = torch.cat([con_input, cat_output], dim=-1)

        # initialize the hidden layer state
        if hidden is None:
            h_0 = x.data.new(self.num_layers, batch_size, self.hidden_size).fill_(0).float()
        else:
            h_0 = hidden
            
        # GRU operation
        output, h_0 = self.gru(x, h_0)

        # obtain the dimension information of the GRU output
        batch_size, timestep, hidden_size = output.shape  
            
        # batch_size * timestep, hidden_dim    
        output = output.reshape(-1, hidden_size)

        output = self.mlp(output)
        
        output = output.reshape(-1, 48, 4)
    
        return output, cat_output
    
# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, feature_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()

        self.hidden_size = hidden_size  # hidden layer size
        self.num_layers = num_layers  # number of LSTM layers

        self.lstm = nn.LSTM(
            feature_size,
            hidden_size,
            num_layers,
            batch_first=True
        )

        # embedding for category variables(station,time,hour)
        self.embedding_layer_station = nn.Embedding(14, 2)
        self.embedding_layer_step = nn.Embedding(48, 2)
        self.embedding_layer_hour = nn.Embedding(24, 2)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )

    def forward(self, con_input, cat_input, hidden=None):
        batch_size = con_input.shape[0]
        
        # embedding for category variables(station,time,hour)
        station_output = self.embedding_layer_station(cat_input[..., 0].long())
        step_output = self.embedding_layer_step(cat_input[..., 1].long())
        hour_output = self.embedding_layer_hour(cat_input[..., 2].long())

        cat_output = torch.cat([station_output, step_output, hour_output], dim=-1)

        x = torch.cat([con_input, cat_output], dim=-1)

        # initialize the hidden layer state
        if hidden is None:
            h_0 = x.data.new(self.num_layers, batch_size, self.hidden_size).fill_(0).float()
            c_0 = x.data.new(self.num_layers, batch_size, self.hidden_size).fill_(0).float()
        else:
            h_0, c_0 = hidden

        # LSTM operation
        output, (h_n, c_n) = self.lstm(x, (h_0, c_0))

        # obtain the dimension information of the LSTM output
        batch_size, timestep, hidden_size = output.shape

        # batch_size * timestep, hidden_dim 
        output = output.reshape(-1, hidden_size)

        output = self.mlp(output)

        output = output.reshape(-1, 48, 4)

        return output, cat_output

# DeepWind Heterogeneous Model
class HeterogeneousModel(nn.Module):
    def __init__(self, num_con_features, feature_size, hidden_size, num_layers, output_size, timestep):
        super(HeterogeneousModel, self).__init__()

        # non-recursive branch
        self.cnn = CNNModel(num_con_features)
        # recursive branch
        self.gru = GRUModel( feature_size, hidden_size, num_layers, output_size)

        # initialize learnable weights
        self.cnn_weight = nn.Parameter(F.softplus(torch.randn(timestep, output_size)), requires_grad = True)
        self.gru_weight = nn.Parameter(F.softplus(torch.randn(timestep, output_size)), requires_grad = True)


    def forward(self, con_input, cat_input, hidden=None):

        cnn_output, cat_output = self.cnn(con_input, cat_input)
        gru_output, cat_output = self.gru(con_input, cat_input, hidden)

        cnn_output = cnn_output * self.cnn_weight
        gru_output = gru_output * self.gru_weight

        # branch fusion
        output = cnn_output+gru_output
      
        return output, cat_output

# custom loss function
class CustomLoss(nn.Module):
    def __init__(self, item_target, quantile):
        super(CustomLoss, self).__init__()
        self.item_target = item_target
      
        self.quantile = quantile

    def forward(self, output, target):

        # calculate feature change trend
        # trend difference loss
        
        # print(output[:,:-1,2:])
        output_diff = output[:,1:,2:] - output[:,:-1,2:]
        target_diff = target[:,1:,2:] - target[:,:-1,2:]

        diff_errors = target_diff - output_diff
        loss_diff = torch.mean(diff_errors).float()
        loss_diff = (loss_diff)**2
        loss_diff = torch.sqrt(loss_diff)
        

        # RMSE loss
        loss = 0
        quantile = self.quantile
        wdir_errors = torch.mean((target[:,:,0:2] - output[:,:,0:2])**2)
        loss += torch.sqrt(wdir_errors)*0.5# 0.5
 
        # quantile loss
        spd_errors = target[:,:,2:] - output[:,:,2:]
        spd_loss = torch.max(quantile*spd_errors, (quantile - 1) * spd_errors)
        loss += torch.mean(spd_loss).float()

        return loss, loss_diff

def train_step(model, train_dl, optimizer, custom_loss, loss_factor):
    train_loss = 0.0
    train_acc = 0.0
    model.train()

    for x1, x2, y in train_dl:
        
        optimizer.zero_grad()
    
        outputs, cat_output = model(x1, x2)

        loss, loss_diff = custom_loss(outputs, y)
    
        loss_total = loss + loss_factor * loss_diff
 
        loss_total.backward()
        optimizer.step()
        train_loss += loss.item() * y.size(0)
        train_acc += loss_diff.item() * y.size(0)
    train_loss = train_loss / len(train_dl.dataset)
    train_acc = train_acc / len(train_dl.dataset)
   
    return train_loss, train_acc

def test_step(model, dl, custom_loss):
    test_loss = 0.0
    test_acc = 0.0
    model.eval()
    
    with torch.no_grad():
        x1, x2, y = dl
        outputs, cat_output = model(x1, x2)
        loss, loss_diff = custom_loss(outputs, y)
        test_loss = loss.item()
        test_acc = loss_diff.item()

    return test_loss, test_acc

# data transformation
def trans_data(data, feats, item_target, cat_feats):
    # data = data.sort_values(['sample', 'station', 'time']).reset_index(drop=True)
    data = data.copy(deep=True).reset_index(drop=True)
    data_x_con = data[feats].values

    data_x_con = data_x_con.reshape((-1,14,48,data_x_con.shape[-1])) # 12
    data_x_cat = data[cat_feats].values
    data_x_cat = data_x_cat.reshape((-1,14,48,data_x_cat.shape[-1])) # 12
    data_y = data[item_target].values
    data_y = data_y.reshape((-1,14,48,4))

    data_x_con = data_x_con.reshape(-1, 48, data_x_con.shape[-1])
    data_x_cat = data_x_cat.reshape(-1, 48, data_x_cat.shape[-1])
    data_y = data_y.reshape(-1, 48, 4)#################change

    # data_x_cat = np.transpose(data_x_cat, (0, 2, 1))
    # data_x_con = np.transpose(data_x_con, (0, 2, 1))

    data_x_con = data_x_con.astype(np.float32)
    data_x_cat = data_x_cat.astype(np.float32)
    return data_x_con, data_x_cat, data_y

# training and prediction of the CNN model
class cv_nn_model():
    def __init__(self, task_name='cnn', nfold=5, seed=None, save_path=None, item_target=None, feats_dict=None, cat_feats=None, device=None, loss_factor=0, quantile=0):
        self.nfold = nfold
        self.seed = seed
        self.save_path = save_path
        self.models = []
        self.task_name = task_name

        self.item_target = item_target
        self.feats_dict = feats_dict
        self.cat_feats = cat_feats

        self.device = device
        self.loss_factor = loss_factor
        self.quantile = quantile
            
    # load model
    def load_model(self):
        self.models = []
        for i in range(self.nfold):
            model = CNNModel(len(self.feats_dict[self.item_target])).to(self.device)
            model.load_state_dict(torch.load(f'{self.save_path}/{self.item_target}_{i}.pth', map_location=self.device))
            self.models.append(model)

    # training without k fold
    def pfit(self, train_data, args):
        fold = 'non_fold'
        # 不使用交叉验证    training without k fold
        train_oof = np.zeros((177*14*48,4))
        dataloader_kwargs = {'num_workers': 8, 'pin_memory': True, 'prefetch_factor': 5} if self.device=='cuda' else {}
        trn_idx = [ti for ti in range(824*14*48)]
        val_idx = [vi for vi in range(824*14*48, (824+177)*14*48)]
        tst_idx = [si for si in range((824+177)*14*48, (824+177+177)*14*48)]
        trn_x_con, trn_x_cat, trn_y = trans_data(train_data.iloc[trn_idx], self.feats_dict[tuple(self.item_target)], self.item_target, self.cat_feats)
        val_x_con, val_x_cat, val_y = trans_data(train_data.iloc[val_idx], self.feats_dict[tuple(self.item_target)], self.item_target, self.cat_feats)
        tst_x_con, tst_x_cat, tst_y = trans_data(train_data.iloc[tst_idx], self.feats_dict[tuple(self.item_target)], self.item_target, self.cat_feats)
        print('train:', trn_x_con.shape, trn_x_cat.shape, trn_y.shape)
        print('val', val_x_con.shape, val_x_cat.shape, val_y.shape)
        print('test', tst_x_con.shape, tst_x_cat.shape, tst_y.shape)


        trn_x_con = torch.from_numpy(trn_x_con).to(self.device)
        trn_x_cat = torch.from_numpy(trn_x_cat).to(self.device)
        trn_y = torch.from_numpy(trn_y).to(self.device)
        

        val_x_con = torch.from_numpy(val_x_con).to(self.device)
        val_x_cat = torch.from_numpy(val_x_cat).to(self.device)
        val_y = torch.from_numpy(val_y).to(self.device)

        tst_x_con = torch.from_numpy(tst_x_con).to(self.device)
        tst_x_cat = torch.from_numpy(tst_x_cat).to(self.device)
        tst_y = torch.from_numpy(tst_y).to(self.device)

        trn_ds = MyDataset(trn_x_con, trn_x_cat, trn_y)
        train_dl = DataLoader(trn_ds, batch_size=1024, shuffle=True, **dataloader_kwargs)#, num_workers=0)
        lr = args.learning_rate

        ########################Choose Model###############################

        # model initialization
        model = CNNModel(trn_x_con.shape[-1]).to(self.device)

        ########################End Choose Model############################

        optimizer = optim.Adam(model.parameters(), lr=lr)

        num_epochs = args.num_epochs
        early_stop = args.early_stop
        early_value = -np.inf

        custom_loss = CustomLoss(self.item_target, self.quantile)

        train_loss = []
        train_acc = []
        test_loss = []
        test_acc = []
        item_train_acc = 0
        item_test_acc = 0
        early_stop_count = 0
        pbar = tqdm(range(num_epochs), desc=f'{self.item_target}_{fold}')
        for epoch in pbar:
            item_train_loss, item_train_acc = train_step(model, train_dl, optimizer, custom_loss, self.loss_factor)
            # print('item_train_loss:',item_train_loss)
            train_loss.append(item_train_loss)
            train_acc.append(item_train_acc)

            item_test_loss, item_test_acc = test_step(model, (val_x_con, val_x_cat, val_y), custom_loss)
            # print('item_test_loss:',item_test_loss)
            test_loss.append(item_test_loss)
            test_acc.append(item_test_acc)

            if -(item_test_loss + self.loss_factor*item_test_acc) > early_value:
                # print('Epoch: {}, Train Loss: {:.4f}, Train Acc: {:.4f}, Test Loss: {:.4f}, Test Acc: {:.4f}'.format(epoch+1, item_train_loss, item_train_acc, item_test_loss, item_test_acc))
                pbar.set_description(f'{self.item_target}_{fold}_epoch{epoch}')
                early_value = -(item_test_loss + self.loss_factor*item_test_acc)
                early_stop_count = 0
                torch.save(model.state_dict(), f'{self.save_path}/{self.item_target}_{fold}.pth')
            else:
                early_stop_count += 1
                if early_stop_count == early_stop:
                    break
        idx = np.argmin(test_loss) 
        print(f'Epoch: {idx}, Train Loss: {train_loss[idx]:.4f}, Train Loss Diff: {train_acc[idx]:.4f}, Test Loss: {test_loss[idx]:.4f}, Test Loss Diff: {test_acc[idx]:.4f}')
        
        model.load_state_dict(torch.load(f'{self.save_path}/{self.item_target}_{fold}.pth'))
        model.eval()
        tst_y_pred, cat_output = model(torch.tensor(tst_x_con).to(self.device), torch.tensor(tst_x_cat).to(self.device))
        tst_y_pred = tst_y_pred.to('cpu').detach().numpy().reshape(-1,4)
        train_oof[[si for si in range((177)*14*48)]] = tst_y_pred
        self.models.append(model)


        return train_oof, model, cat_output

    # predict
    def predict(self, test_data):
        test_pred = np.zeros(len(test_data))
        tet_x_con, tet_x_cat, tet_y = trans_data(test_data, self.feats_dict[self.item_target], self.item_target, self.cat_feats)
        for i, model in enumerate(self.models):
            model.eval()
            test_pred += model(torch.tensor(tet_x_con).to(self.device), torch.tensor(tet_x_cat).to(self.device)).to('cpu').detach().numpy().reshape(-1) / 5
        return test_pred

# training and prediction of the GRU model
class gru_nn_model():
    def __init__(self, task_name='gru', nfold=5, seed=None, save_path=None, item_target=None, feats_dict=None, cat_feats=None,  device=None, loss_factor=0, quantile=0):
        self.nfold = nfold
        self.seed = seed
        self.save_path = save_path
        self.models = []
        self.task_name = task_name

        self.item_target = item_target
        self.feats_dict = feats_dict
        self.cat_feats = cat_feats

        self.device = device
        self.loss_factor = loss_factor
        self.quantile = quantile
            
    # load model
    def load_model(self):
        self.models = []
        for i in range(self.nfold):
            model = GRUModel(len(self.feats_dict[self.item_target])).to(self.device)
            model.load_state_dict(torch.load(f'{self.save_path}/{self.item_target}_{i}.pth', map_location=self.device))
            self.models.append(model)

    # training without k fold
    def pfit(self, train_data, args):
        fold = 'non_fold'
        # training without k fold
        train_oof = np.zeros((177*14*48,4))
        dataloader_kwargs = {'num_workers': 8, 'pin_memory': True, 'prefetch_factor': 5} if self.device=='cuda' else {}
        trn_idx = [ti for ti in range(824*14*48)]
        val_idx = [vi for vi in range(824*14*48, (824+177)*14*48)]
        tst_idx = [si for si in range((824+177)*14*48, (824+177+177)*14*48)]
        trn_x_con, trn_x_cat, trn_y = trans_data(train_data.iloc[trn_idx], self.feats_dict[tuple(self.item_target)], self.item_target, self.cat_feats)
        val_x_con, val_x_cat, val_y = trans_data(train_data.iloc[val_idx], self.feats_dict[tuple(self.item_target)], self.item_target, self.cat_feats)
        tst_x_con, tst_x_cat, tst_y = trans_data(train_data.iloc[tst_idx], self.feats_dict[tuple(self.item_target)], self.item_target, self.cat_feats)
        print('train:', trn_x_con.shape, trn_x_cat.shape, trn_y.shape)
        print('val', val_x_con.shape, val_x_cat.shape, val_y.shape)
        print('test', tst_x_con.shape, tst_x_cat.shape, tst_y.shape)


        trn_x_con = torch.from_numpy(trn_x_con).to(self.device)
        trn_x_cat = torch.from_numpy(trn_x_cat).to(self.device)
        trn_y = torch.from_numpy(trn_y).to(self.device)
        

        val_x_con = torch.from_numpy(val_x_con).to(self.device)
        val_x_cat = torch.from_numpy(val_x_cat).to(self.device)
        val_y = torch.from_numpy(val_y).to(self.device)

        tst_x_con = torch.from_numpy(tst_x_con).to(self.device)
        tst_x_cat = torch.from_numpy(tst_x_cat).to(self.device)
        tst_y = torch.from_numpy(tst_y).to(self.device)

        trn_ds = MyDataset(trn_x_con, trn_x_cat, trn_y)
        train_dl = DataLoader(trn_ds, batch_size=512, shuffle=True, **dataloader_kwargs)#, num_workers=0)
        lr = args.learning_rate

        ########################Choose Model###############################

        # setting and initializing model parameters
        feature_size = args.feature_size  # the number of features for each stepe
        hidden_size = args.hidden_size  # hidden layer size
        output_size = args.output_size  # Predict 4 targets
        num_layers = args.num_layers  # Number of layers in the GRU
        model = GRUModel(feature_size, hidden_size, num_layers, output_size).to(self.device)

        ########################End Choose Model############################

        optimizer = optim.Adam(model.parameters(), lr=lr)

        num_epochs = args.num_epochs
        early_stop = args.early_stop
        early_value = -np.inf

        custom_loss = CustomLoss(self.item_target, self.quantile)

        train_loss = []
        train_acc = []
        test_loss = []
        test_acc = []
        item_train_acc = 0
        item_test_acc = 0
        early_stop_count = 0
        pbar = tqdm(range(num_epochs), desc=f'{self.item_target}_{fold}')
        for epoch in pbar:
            item_train_loss, item_train_acc = train_step(model, train_dl, optimizer, custom_loss, self.loss_factor)
            # print('item_train_loss:',item_train_loss)
            train_loss.append(item_train_loss)
            train_acc.append(item_train_acc)

            item_test_loss, item_test_acc = test_step(model, (val_x_con, val_x_cat, val_y), custom_loss)
            # print('item_test_loss:',item_test_loss)
            test_loss.append(item_test_loss)
            test_acc.append(item_test_acc)

            if -(item_test_loss + self.loss_factor*item_test_acc) > early_value:
                # print('Epoch: {}, Train Loss: {:.4f}, Train Acc: {:.4f}, Test Loss: {:.4f}, Test Acc: {:.4f}'.format(epoch+1, item_train_loss, item_train_acc, item_test_loss, item_test_acc))
                pbar.set_description(f'{self.item_target}_{fold}_epoch{epoch}')
                early_value = -(item_test_loss + self.loss_factor*item_test_acc)
                early_stop_count = 0
                torch.save(model.state_dict(), f'{self.save_path}/{self.item_target}_{fold}.pth')
            else:
                early_stop_count += 1
                if early_stop_count == early_stop:
                    break
        idx = np.argmin(test_loss) 
        print(f'Epoch: {idx}, Train Loss: {train_loss[idx]:.4f}, Train Loss Diff: {train_acc[idx]:.4f}, Test Loss: {test_loss[idx]:.4f}, Test Loss Diff: {test_acc[idx]:.4f}')
        
        model.load_state_dict(torch.load(f'{self.save_path}/{self.item_target}_{fold}.pth'))
        model.eval()
        tst_y_pred, cat_output = model(torch.tensor(tst_x_con).to(self.device), torch.tensor(tst_x_cat).to(self.device))
        tst_y_pred = tst_y_pred.to('cpu').detach().numpy().reshape(-1,4)
        train_oof[[si for si in range((177)*14*48)]] = tst_y_pred
        self.models.append(model)


        return train_oof, model, cat_output

    # predict
    def predict(self, test_data):
        test_pred = np.zeros(len(test_data))
        tet_x_con, tet_x_cat, tet_y = trans_data(test_data, self.feats_dict[self.item_target], self.item_target, self.cat_feats)
        for i, model in enumerate(self.models):
            model.eval()
            test_pred += model(torch.tensor(tet_x_con).to(self.device), torch.tensor(tet_x_cat).to(self.device)).to('cpu').detach().numpy().reshape(-1) / 5
        return test_pred
    
# training and prediction of the LSTM model
class lstm_nn_model():
    def __init__(self, task_name='lstm', nfold=5, seed=None, save_path=None, item_target=None, feats_dict=None, cat_feats=None,  device=None, loss_factor=0, quantile=0):
        self.nfold = nfold
        self.seed = seed
        self.save_path = save_path
        self.models = []
        self.task_name = task_name

        self.item_target = item_target
        self.feats_dict = feats_dict
        self.cat_feats = cat_feats

        self.device = device
        self.loss_factor = loss_factor
        self.quantile = quantile
            
    # load model
    def load_model(self):
        self.models = []
        for i in range(self.nfold):
            model = LSTMModel(len(self.feats_dict[self.item_target])).to(self.device)
            model.load_state_dict(torch.load(f'{self.save_path}/{self.item_target}_{i}.pth', map_location=self.device))
            self.models.append(model)

    # training without k fold
    def pfit(self, train_data, args):
        fold = 'non_fold'
        # training without k fold
        train_oof = np.zeros((177*14*48,4))
        dataloader_kwargs = {'num_workers': 8, 'pin_memory': True, 'prefetch_factor': 5} if self.device=='cuda' else {}
        trn_idx = [ti for ti in range(824*14*48)]
        val_idx = [vi for vi in range(824*14*48, (824+177)*14*48)]
        tst_idx = [si for si in range((824+177)*14*48, (824+177+177)*14*48)]
        trn_x_con, trn_x_cat, trn_y = trans_data(train_data.iloc[trn_idx], self.feats_dict[tuple(self.item_target)], self.item_target, self.cat_feats)
        val_x_con, val_x_cat, val_y = trans_data(train_data.iloc[val_idx], self.feats_dict[tuple(self.item_target)], self.item_target, self.cat_feats)
        tst_x_con, tst_x_cat, tst_y = trans_data(train_data.iloc[tst_idx], self.feats_dict[tuple(self.item_target)], self.item_target, self.cat_feats)
        print('train:', trn_x_con.shape, trn_x_cat.shape, trn_y.shape)
        print('val', val_x_con.shape, val_x_cat.shape, val_y.shape)
        print('test', tst_x_con.shape, tst_x_cat.shape, tst_y.shape)


        trn_x_con = torch.from_numpy(trn_x_con).to(self.device)
        trn_x_cat = torch.from_numpy(trn_x_cat).to(self.device)
        trn_y = torch.from_numpy(trn_y).to(self.device)
        

        val_x_con = torch.from_numpy(val_x_con).to(self.device)
        val_x_cat = torch.from_numpy(val_x_cat).to(self.device)
        val_y = torch.from_numpy(val_y).to(self.device)

        tst_x_con = torch.from_numpy(tst_x_con).to(self.device)
        tst_x_cat = torch.from_numpy(tst_x_cat).to(self.device)
        tst_y = torch.from_numpy(tst_y).to(self.device)

        trn_ds = MyDataset(trn_x_con, trn_x_cat, trn_y)
        train_dl = DataLoader(trn_ds, batch_size=512, shuffle=True, **dataloader_kwargs)#, num_workers=0)
        lr = args.learning_rate

        ########################Choose Model###############################

        #setting and initializing model parameters
        feature_size = args.feature_size  # the number of features for each stepe
        hidden_size = args.hidden_size  # hidden layer size
        output_size = args.output_size  # Predict 4 targets
        num_layers = args.numlayers  # Number of layers in the LSTM
        model = LSTMModel(feature_size, hidden_size, num_layers, output_size).to(self.device)

        ########################End Choose Model############################

        optimizer = optim.Adam(model.parameters(), lr=lr)

        num_epochs = args.num_epochs
        early_stop = args.early_stop
        early_value = -np.inf

        custom_loss = CustomLoss(self.item_target, self.quantile)

        train_loss = []
        train_acc = []
        test_loss = []
        test_acc = []
        item_train_acc = 0
        item_test_acc = 0
        early_stop_count = 0
        pbar = tqdm(range(num_epochs), desc=f'{self.item_target}_{fold}')
        for epoch in pbar:
            item_train_loss, item_train_acc = train_step(model, train_dl, optimizer, custom_loss, self.loss_factor)
            # print('item_train_loss:',item_train_loss)
            train_loss.append(item_train_loss)
            train_acc.append(item_train_acc)

            item_test_loss, item_test_acc = test_step(model, (val_x_con, val_x_cat, val_y), custom_loss)
            # print('item_test_loss:',item_test_loss)
            test_loss.append(item_test_loss)
            test_acc.append(item_test_acc)

            if -(item_test_loss + self.loss_factor*item_test_acc) > early_value:
                # print('Epoch: {}, Train Loss: {:.4f}, Train Acc: {:.4f}, Test Loss: {:.4f}, Test Acc: {:.4f}'.format(epoch+1, item_train_loss, item_train_acc, item_test_loss, item_test_acc))
                pbar.set_description(f'{self.item_target}_{fold}_epoch{epoch}')
                early_value = -(item_test_loss + self.loss_factor*item_test_acc)
                early_stop_count = 0
                torch.save(model.state_dict(), f'{self.save_path}/{self.item_target}_{fold}.pth')
            else:
                early_stop_count += 1
                if early_stop_count == early_stop:
                    break
        idx = np.argmin(test_loss) 
        print(f'Epoch: {idx}, Train Loss: {train_loss[idx]:.4f}, Train Loss Diff: {train_acc[idx]:.4f}, Test Loss: {test_loss[idx]:.4f}, Test Loss Diff: {test_acc[idx]:.4f}')
        
        model.load_state_dict(torch.load(f'{self.save_path}/{self.item_target}_{fold}.pth'))
        model.eval()
        tst_y_pred, cat_output = model(torch.tensor(tst_x_con).to(self.device), torch.tensor(tst_x_cat).to(self.device))
        tst_y_pred = tst_y_pred.to('cpu').detach().numpy().reshape(-1,4)
        train_oof[[si for si in range((177)*14*48)]] = tst_y_pred
        self.models.append(model)


        return train_oof, model, cat_output

    # predict
    def predict(self, test_data):
        test_pred = np.zeros(len(test_data))
        tet_x_con, tet_x_cat, tet_y = trans_data(test_data, self.feats_dict[self.item_target], self.item_target, self.cat_feats)
        for i, model in enumerate(self.models):
            model.eval()
            test_pred += model(torch.tensor(tet_x_con).to(self.device), torch.tensor(tet_x_cat).to(self.device)).to('cpu').detach().numpy().reshape(-1) / 5
        return test_pred
    
# training and prediction of the heterogeneous model
class heterogeneous_nn_model():
    def __init__(self, task_name='heterogeneous', nfold=5, seed=None, save_path=None, item_target=None, feats_dict=None, cat_feats=None, device=None, loss_factor=0, quantile=0):
        self.nfold = nfold
        self.seed = seed
        self.save_path = save_path
        self.models = []
        self.task_name = task_name

        self.item_target = item_target
        self.feats_dict = feats_dict
        self.cat_feats = cat_feats
        
        self.device = device
        self.loss_factor = loss_factor
        self.quantile = quantile
    
    # load model
    def load_model(self):
        self.models = []
        for i in range(self.nfold):
            model = CNNModel(len(self.feats_dict[self.item_target])).to(self.device)
            model.load_state_dict(torch.load(f'{self.save_path}/{self.item_target}_{i}.pth', map_location=self.device))
            self.models.append(model)

    # training without k fold
    def pfit(self, train_data, args):
        fold = 'non_fold'
        # training without k fold
        train_oof = np.zeros((177*14*48,4))
        dataloader_kwargs = {'num_workers': 8, 'pin_memory': True, 'prefetch_factor': 5} if self.device=='cuda' else {}
        trn_idx = [ti for ti in range(824*14*48)]
        val_idx = [vi for vi in range(824*14*48, (824+177)*14*48)]
        tst_idx = [si for si in range((824+177)*14*48, (824+177+177)*14*48)]
        
        trn_x_con, trn_x_cat, trn_y = trans_data(train_data.iloc[trn_idx], self.feats_dict[tuple(self.item_target)], self.item_target, self.cat_feats)
        val_x_con, val_x_cat, val_y = trans_data(train_data.iloc[val_idx], self.feats_dict[tuple(self.item_target)], self.item_target, self.cat_feats)
        tst_x_con, tst_x_cat, tst_y = trans_data(train_data.iloc[tst_idx], self.feats_dict[tuple(self.item_target)], self.item_target, self.cat_feats)
        print('train:', trn_x_con.shape, trn_x_cat.shape, trn_y.shape)
        print('val', val_x_con.shape, val_x_cat.shape, val_y.shape)
        print('test', tst_x_con.shape, tst_x_cat.shape, tst_y.shape)


        trn_x_con = torch.from_numpy(trn_x_con).to(self.device)
        trn_x_cat = torch.from_numpy(trn_x_cat).to(self.device)
        trn_y = torch.from_numpy(trn_y).to(self.device)
        

        val_x_con = torch.from_numpy(val_x_con).to(self.device)
        val_x_cat = torch.from_numpy(val_x_cat).to(self.device)
        val_y = torch.from_numpy(val_y).to(self.device)

        tst_x_con = torch.from_numpy(tst_x_con).to(self.device)
        tst_x_cat = torch.from_numpy(tst_x_cat).to(self.device)
        tst_y = torch.from_numpy(tst_y).to(self.device)

        trn_ds = MyDataset(trn_x_con, trn_x_cat, trn_y)
        train_dl = DataLoader(trn_ds, batch_size=4096, **dataloader_kwargs)#, num_workers=0)
        lr = args.learning_rate

        ########################Choose Model###############################

        # Parameter setting for recursive branching (GRU model)
        feature_size = args.feature_size  # the number of features for each stepe
        hidden_size = args.hidden_size  # hidden layer size
        output_size = args.output_size  # Predict 4 targets
        num_layers = args.num_layers # Number of layers in the GRU
        timestep = args.timestep

        #model initialization
        model = HeterogeneousModel(trn_x_con.shape[-1], feature_size, hidden_size, num_layers, output_size, timestep).to(self.device)
        
        ########################End Choose Model############################

        optimizer = optim.Adam(model.parameters(), lr=lr)

        num_epochs = args.num_epochs
        early_stop = args.early_stop
        early_value = -np.inf

        custom_loss = CustomLoss(self.item_target, self.quantile)

        train_loss = []
        train_acc = []
        test_loss = []
        test_acc = []
        item_train_acc = 0
        item_test_acc = 0
        early_stop_count = 0
        pbar = tqdm(range(num_epochs), desc=f'{self.item_target}_{fold}')
        for epoch in pbar:
            item_train_loss, item_train_acc = train_step(model, train_dl, optimizer, custom_loss, self.loss_factor)
            # print('item_train_loss:',item_train_loss)
            train_loss.append(item_train_loss)
            train_acc.append(item_train_acc)

            item_test_loss, item_test_acc = test_step(model, (val_x_con, val_x_cat, val_y), custom_loss)
            # print('item_test_loss:',item_test_loss)
            test_loss.append(item_test_loss)
            test_acc.append(item_test_acc)

            if -(item_test_loss + self.loss_factor*item_test_acc) > early_value:
                # print('Epoch: {}, Train Loss: {:.4f}, Train Acc: {:.4f}, Test Loss: {:.4f}, Test Acc: {:.4f}'.format(epoch+1, item_train_loss, item_train_acc, item_test_loss, item_test_acc))
                pbar.set_description(f'{self.item_target}_{fold}_epoch{epoch}')
                early_value = -(item_test_loss + self.loss_factor*item_test_acc)
                early_stop_count = 0
                torch.save(model.state_dict(), f'{self.save_path}/{self.item_target}_{fold}.pth')
            else:
                early_stop_count += 1
                if early_stop_count == early_stop:
                    break
        idx = np.argmin(test_loss) 
        print(f'Epoch: {idx}, Train Loss: {train_loss[idx]:.4f}, Train Loss Diff: {train_acc[idx]:.4f}, Test Loss: {test_loss[idx]:.4f}, Test Loss Diff: {test_acc[idx]:.4f}')
        
        model.load_state_dict(torch.load(f'{self.save_path}/{self.item_target}_{fold}.pth'))
        model.eval()
        tst_y_pred, cat_output = model(torch.tensor(tst_x_con).to(self.device), torch.tensor(tst_x_cat).to(self.device))
        tst_y_pred = tst_y_pred.to('cpu').detach().numpy().reshape(-1,4)
        train_oof[[si for si in range((177)*14*48)]] = tst_y_pred
        self.models.append(model)


        return train_oof, model, cat_output

    # predict
    def predict(self, test_data):
        test_pred = np.zeros(len(test_data))
        tet_x_con, tet_x_cat, tet_y = trans_data(test_data, self.feats_dict[self.item_target], self.item_target, self.cat_feats)
        for i, model in enumerate(self.models):
            model.eval()
            test_pred += model(torch.tensor(tet_x_con).to(self.device), torch.tensor(tet_x_cat).to(self.device)).to('cpu').detach().numpy().reshape(-1) / 5
        return test_pred

# select model
def select_model(type, task_name, nfold, seed, save_path, item_target, feats_dict, cat_feats, device, loss_factor, quantile):
    if(type == 'cnn'):
        return cv_nn_model(task_name, nfold, seed, save_path, item_target, feats_dict, cat_feats, device, loss_factor, quantile)
    elif(type == 'gru'):
        return gru_nn_model(task_name, nfold, seed, save_path, item_target, feats_dict, cat_feats, device, loss_factor, quantile)
    elif(type == 'lstm'):
        return lstm_nn_model(task_name, nfold, seed, save_path, item_target, feats_dict, cat_feats, device, loss_factor, quantile)
    else:
        return heterogeneous_nn_model(task_name, nfold, seed, save_path, item_target, feats_dict, cat_feats, device, loss_factor, quantile)
    