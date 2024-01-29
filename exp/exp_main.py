import os
import warnings
import numpy as np
import pandas as pd

import random
import torch
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic=True

warnings.filterwarnings("ignore")

from cyeva import WindComparison

from exp.basemodel import utils

warnings.filterwarnings('ignore')


class Exp_Main(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.seed = self._acquire_seed()
        self.targets = self._get_targets()
        self.train_data, self.scaler, self.target_scaler, self.feats_dict, self.cat_feats = self._data_preprocessing()
        self.model = self._build_model()
    
    def _get_targets(self):
        targets = self.args.TARGET_FEATS.split(',')
        return  targets
        
    def _acquire_device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _acquire_seed(self):
        SEED = 2023
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED) 
        torch.cuda.manual_seed_all(SEED)
        return SEED
    
    def _data_preprocessing(self):
       
        if not os.path.exists(self.args.MODEL_SAVE_PATH):
            os.makedirs(self.args.MODEL_SAVE_PATH)
        if not os.path.exists(self.args.RESULT_SAVE_PATH):
            os.makedirs(self.args.RESULT_SAVE_PATH)
        cat_feats = []
        
        train_data = pd.read_pickle(os.path.join(self.args.DATA_PATH, self.args.dataset))
        
        train_data['type'] = 'train'
        data = train_data

        # Remove the problematic target.
        # print(self.targets)
        for item in self.targets:
            idx = data[item] >= 199999.0
            data.loc[idx, item] = np.nan

        if data.isnull().sum().sum() != 0:
            data = data.ffill().bfill()  

        # Convert wind direction
        data.insert(1, 'cos', np.cos(data[self.targets[0]]/360*2*np.pi))
        data.insert(1, 'sin', np.sin(data[self.targets[0]]/360*2*np.pi))
        
        self.targets.pop(0) 

        self.targets.insert(0, 'cos')  
        self.targets.insert(0, 'sin')  
        
        data, cat_feats = utils.gen_feats(data)
        
        feats = [item for item in data.columns if item not in self.targets+['ID', 'sample', 'type', 'real_sample', 'wdir_2min']] # 83
        con_feats = [item for item in feats if item not in cat_feats] # 80
        
        feats_dict = {
            self.targets[0]: [item for item in con_feats if not item.endswith('spd')],
            self.targets[1]: [item for item in con_feats if not item.endswith('spd')],
            self.targets[2]: [item for item in con_feats if not item.endswith('dir')],
            self.targets[3]: [item for item in con_feats if not item.endswith('dir')],
            (self.targets[0], self.targets[1], self.targets[2], self.targets[3]):[item for item in con_feats],
        }
        # for item in feats_dict:
        #     print(item, len(feats_dict[item]), feats_dict[item])
            
        train_data = data.query('type=="train"').reset_index(drop=True)
        train_data = train_data.sort_values(['sample', 'station']).reset_index(drop=True)
        
        # train_data.to_csv('dataset/pre_train_data.csv')

        # group_x = train_data['sample'].reset_index(drop=True)
        # data_dir = data[['ID', '10_dir']].copy(deep=True)
        
        # train_data = pd.read_csv('dataset/pre_train_data.csv')
        
        ss = [si for si in range((824+177)*14*48, (824+177+177)*14*48)]
        
        # columns = ['ID', 'wdir_2min', 'spd_2min', 'spd_inst_max']

        # DataFrame
        data = {
            'ID': train_data.iloc[ss]['ID'],
            'wdir_2min': train_data.iloc[ss]['10_dir'],
            'spd_2min': train_data.iloc[ss]['10_spd'],
            'spd_inst_max': train_data.iloc[ss]['10_spd']
        }
        
        train_data, scaler, target_scaler = self._maxmin_scaler(train_data, con_feats)        
        
        train_data[self.targets] = train_data[self.targets].astype('float32')
        
        return train_data, scaler, target_scaler, feats_dict, cat_feats
        
        
    def _standard_scaler(self, data, con_feats):  
        std = StandardScaler()
        std.fit(data[con_feats])
        data[con_feats] = std.transform(data[con_feats])

        target_std = StandardScaler()
        target_std.fit(data[self.targets])
        data[self.targets] = target_std.transform(data[self.targets])
        
        return data, std, target_std
        
    def _maxmin_scaler(self, data, con_feats):
        minmax = MinMaxScaler()
        minmax.fit(data[con_feats])
        data[con_feats] = minmax.transform(data[con_feats])

        target_minmax = MinMaxScaler()
        target_minmax.fit(data[self.targets])
        data[self.targets] = target_minmax.transform(data[self.targets])
        
        return data, minmax, target_minmax
         
    def _build_model(self):
        
        model_dict = {
            'HeterogeneousModel': 'heterogeneous',
            'CNNModel': 'cnn',
            'GRUModel': 'gru',
            'LSTMModel': 'lstm'
        }
        
        model_type = model_dict[self.args.model]
        
        model = utils.select_model(model_type, self.args.model, self.args.nfold, self.seed, self.args.MODEL_SAVE_PATH, self.targets, 
                                   self.feats_dict, self.cat_feats, self.device, self.args.LOSSFACTOR, self.args.QUANTILE)
        
        return model

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def run(self):
        # print(self.targets)
        print("device: ", self.device)
        print('================== start training ==================')
        
        item_oof, model, cat_output = self.model.pfit(self.train_data, self.args)
        test_y = self.train_data.iloc[[si for si in range((824+177)*14*48, (824+177+177)*14*48)]][self.targets]
        test_id = self.train_data.iloc[[si for si in range((824+177)*14*48, (824+177+177)*14*48)]]['ID']
        
        test_pred = pd.DataFrame(item_oof, columns=self.targets)
        test_pred.insert(0, 'ID', test_id.reset_index(drop=True))
        
        test_pred[self.targets] = self.target_scaler.inverse_transform(test_pred[self.targets])
        test_pred['wdir_2min'] = utils.transform_wdir(test_pred['sin'], test_pred['cos'])
        test_pred = test_pred[['ID', 'wdir_2min', 'spd_2min', 'spd_inst_max']]
        
        test_label = pd.DataFrame(test_y.reset_index(drop=True), columns=['sin', 'cos', 'spd_2min', 'spd_inst_max'])
        test_label.insert(0, 'ID', test_id.reset_index(drop=True))
        test_label[self.targets] = self.target_scaler.inverse_transform(test_label[self.targets])
        test_label['wdir_2min'] = utils.transform_wdir(test_label['sin'], test_label['cos'])
        test_label = test_label[['ID', 'wdir_2min', 'spd_2min', 'spd_inst_max']]
        
        print('\t\twdir_2min\tspd_2min\tspd_inst_max')

        print("MSE:\t", self.get_MSE_Score(test_pred.iloc[:,1:],test_label.iloc[:,1:]))
        print("MAE:\t", self.get_MAE_Score(test_pred.iloc[:,1:],test_label.iloc[:,1:]))
        print("SMAPE:\t", self.get_SMAPE_Score(test_pred.iloc[:,1:],test_label.iloc[:,1:]))
        print("IPA:\t", self.get_IPA(test_pred.iloc[:,1:],test_label.iloc[:,1:]))
        
        wind_score, spd_score, spd_max_score = self.wind_spd_score(test_label=test_label, test_pred=test_pred)
        print('score:\t\t', wind_score, '\t\t\t', spd_score, '\t\t', spd_max_score)
        
    
    # metric 
    def wind_spd_score(self, test_label, test_pred):
        wind = WindComparison(
            obs_spd = np.array(test_label.loc[:,'spd_2min']), 
            fct_spd = np.array(test_pred.loc[:,'spd_2min']), 
            obs_dir = np.array(test_label.loc[:,'wdir_2min']), 
            fct_dir = np.array(test_pred.loc[:,'wdir_2min'])
            )
        max_wind = WindComparison(
            obs_spd = np.array(test_label.loc[:,'spd_inst_max']), 
            fct_spd = np.array(test_pred.loc[:,'spd_inst_max']), 
            obs_dir = np.array(test_label.loc[:,'wdir_2min']), 
            fct_dir = np.array(test_pred.loc[:,'wdir_2min'])
            )
        return wind.calc_dir_score(), wind.calc_speed_score(), max_wind.calc_speed_score()

    def get_MSE_Score(self, y_pred,y_true):
        wdir_mse = ((y_pred.iloc[:, 0] - y_true.iloc[:, 0]) ** 2).mean()
        spd_mse = ((y_pred.iloc[:, 1] - y_true.iloc[:, 1]) ** 2).mean()
        maxspd_mse = ((y_pred.iloc[:, 2] - y_true.iloc[:, 2]) ** 2).mean()

        return  wdir_mse, spd_mse, maxspd_mse

    def get_MAE_Score(self, y_pred,y_true):
        wdir_mae = ((y_pred.iloc[:, 0] - y_true.iloc[:, 0]).abs()).mean()
        spd_mae = ((y_pred.iloc[:, 1] - y_true.iloc[:, 1]).abs()).mean()
        maxspd_mae = ((y_pred.iloc[:, 2] - y_true.iloc[:, 2]).abs()).mean()

        return wdir_mae, spd_mae, maxspd_mae

    def get_SMAPE_Score(self, y_pred,y_true):
        wdir_smape = ((y_pred.iloc[:, 0] - y_true.iloc[:, 0]).abs()/((y_true.iloc[:, 0].abs()+y_pred.iloc[:, 0].abs())/2)).mean()
        spd_smape = ((y_pred.iloc[:, 1] - y_true.iloc[:, 1]).abs()/((y_true.iloc[:, 1].abs()+y_pred.iloc[:, 1].abs())/2)).mean()
        maxspd_smape = ((y_pred.iloc[:, 2] - y_true.iloc[:, 2]).abs()/((y_true.iloc[:, 2].abs()+y_pred.iloc[:, 2].abs())/2)).mean()

        return wdir_smape, spd_smape, maxspd_smape 

    def wdir_IPA(self, y_pred,y_true):
        bins = [0, 22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5, 9999]
        labels = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'N']

        true = pd.cut(x=y_true, bins=bins, labels=labels, include_lowest=True, right=True, ordered=False).astype('object')
        pred = pd.cut(x=y_pred, bins=bins, labels=labels, include_lowest=True, right=True, ordered=False).astype('object')

        results = true==pred
        results[true=='C'] = True
        return results.mean()

    def spd_IPA(self, y_pred,y_true):
        bins = [0, 5.5, 8, 10.8, 13.9, 17.2, 9999] # Classification of average wind force inspection levels
        labels = [1, 2, 3, 4, 5, 6] # Classification after grading

        true = pd.cut(x=y_true, bins=bins, labels=labels, include_lowest=True, right=True, ordered=False).astype('object')
        pred = pd.cut(x=y_pred, bins=bins, labels=labels, include_lowest=True, right=True, ordered=False).astype('object')

        idx = y_true<=0.2
        true[idx] = 'C'
        idx = y_pred<=0.2
        pred[idx] = 'C'

        results = true==pred
        results[true=='C'] = True
        return results.mean()

    def max_spd_IPA(self, y_pred,y_true):
        bins = [0, 8, 10.8, 13.9, 17.2, 20.8, 9999] # Classification of extreme wind force inspection levels
        labels = [1, 2, 3, 4, 5, 6] # Classification after grading
        true = pd.cut(x=y_true, bins=bins, labels=labels, include_lowest=True, right=True, ordered=False).astype('object')
        pred = pd.cut(x=y_pred, bins=bins, labels=labels, include_lowest=True, right=True, ordered=False).astype('object')

        idx = y_true<=0.2
        true[idx] = 'C'
        idx = y_pred<=0.2
        pred[idx] = 'C'

        results = true==pred
        results[true=='C'] = True
        return results.mean()

    def get_IPA(self, y_pred,y_true):
        wdir_IPA = self.wdir_IPA(y_pred['wdir_2min'],y_true['wdir_2min'])
        spd_IPA = self.spd_IPA(y_pred['spd_2min'],y_true['spd_2min'])
        maxspd_IPA = self.max_spd_IPA(y_pred['spd_inst_max'],y_true['spd_inst_max'])

        return wdir_IPA, spd_IPA, maxspd_IPA    

