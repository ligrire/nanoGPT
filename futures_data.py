import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

RET_MEAN_STD = 0.00033422861928492277, 2.0654060785101094
PREMIUM_MEAN_STD = -130.87443697806327, 211.71141822708947
PREMIUM_DIFF_MEAN_STD = 0.0004550047121431646, 2.0737513824379423
DATA_DIR = '/Users/shitiancheng/Downloads/futures_data'

FUTURE_DICT = {'IH': 0, 'IF': 1, 'IC': 2, 'IM': 3}

ihs = [('IH0'+ str(i)).encode() for i in range(1, 5)],  b'000016'
ics = [('IC0'+ str(i)).encode() for i in range(1, 5)], b'000905'
ifs = [('IF0'+ str(i)).encode() for i in range(1, 5)],  b'000300'
ims = [('IM0'+ str(i)).encode() for i in range(1, 5)],  b'000852'
CHANGE_MONTH = ['01', '04', '07', '10']
delivery_date = pd.read_pickle(os.path.join(DATA_DIR, 'delivery.pkl'))
DELIVERY_DATE = delivery_date[0].values

PAIRS= [(b'IC01', b'IC02'),
 (b'IC01', b'IC03'),
 (b'IC01', b'IF02'),
 (b'IC01', b'IF04'),
 (b'IC01', b'IH02'),
 (b'IC01', b'IH03'),
 (b'IC01', b'IH04'),
 (b'IC01', b'IM01'),
 (b'IC01', b'IM03'),
 (b'IC01', b'IM04'),
 (b'IC02', b'IC03'),
 (b'IC02', b'IH03'),
 (b'IC02', b'IH04'),
 (b'IC02', b'IM03'),
 (b'IC02', b'IM04'),
 (b'IC03', b'IH04'),
 (b'IC03', b'IM03'),
 (b'IC03', b'IM04'),
 (b'IC04', b'IC01'),
 (b'IC04', b'IC02'),
 (b'IC04', b'IC03'),
 (b'IC04', b'IF01'),
 (b'IC04', b'IF02'),
 (b'IC04', b'IF04'),
 (b'IC04', b'IH02'),
 (b'IC04', b'IH03'),
 (b'IC04', b'IH04'),
 (b'IC04', b'IM01'),
 (b'IC04', b'IM03'),
 (b'IC04', b'IM04'),
 (b'IF01', b'IC01'),
 (b'IF01', b'IC02'),
 (b'IF01', b'IC03'),
 (b'IF01', b'IF02'),
 (b'IF01', b'IF04'),
 (b'IF01', b'IH02'),
 (b'IF01', b'IH03'),
 (b'IF01', b'IH04'),
 (b'IF01', b'IM01'),
 (b'IF01', b'IM03'),
 (b'IF01', b'IM04'),
 (b'IF02', b'IC02'),
 (b'IF02', b'IC03'),
 (b'IF02', b'IH03'),
 (b'IF02', b'IH04'),
 (b'IF02', b'IM03'),
 (b'IF02', b'IM04'),
 (b'IF03', b'IC01'),
 (b'IF03', b'IC02'),
 (b'IF03', b'IC03'),
 (b'IF03', b'IC04'),
 (b'IF03', b'IF01'),
 (b'IF03', b'IF02'),
 (b'IF03', b'IF04'),
 (b'IF03', b'IH01'),
 (b'IF03', b'IH02'),
 (b'IF03', b'IH03'),
 (b'IF03', b'IH04'),
 (b'IF03', b'IM01'),
 (b'IF03', b'IM02'),
 (b'IF03', b'IM03'),
 (b'IF03', b'IM04'),
 (b'IF04', b'IC02'),
 (b'IF04', b'IC03'),
 (b'IF04', b'IF02'),
 (b'IF04', b'IH02'),
 (b'IF04', b'IH03'),
 (b'IF04', b'IH04'),
 (b'IF04', b'IM03'),
 (b'IF04', b'IM04'),
 (b'IH01', b'IC01'),
 (b'IH01', b'IC02'),
 (b'IH01', b'IC03'),
 (b'IH01', b'IC04'),
 (b'IH01', b'IF01'),
 (b'IH01', b'IF02'),
 (b'IH01', b'IF04'),
 (b'IH01', b'IH02'),
 (b'IH01', b'IH03'),
 (b'IH01', b'IH04'),
 (b'IH01', b'IM01'),
 (b'IH01', b'IM02'),
 (b'IH01', b'IM03'),
 (b'IH01', b'IM04'),
 (b'IH02', b'IC02'),
 (b'IH02', b'IC03'),
 (b'IH02', b'IF02'),
 (b'IH02', b'IH03'),
 (b'IH02', b'IH04'),
 (b'IH02', b'IM03'),
 (b'IH02', b'IM04'),
 (b'IH03', b'IC03'),
 (b'IH03', b'IH04'),
 (b'IH03', b'IM03'),
 (b'IH03', b'IM04'),
 (b'IH04', b'IM04'),
 (b'IM01', b'IC02'),
 (b'IM01', b'IC03'),
 (b'IM01', b'IF02'),
 (b'IM01', b'IF04'),
 (b'IM01', b'IH02'),
 (b'IM01', b'IH03'),
 (b'IM01', b'IH04'),
 (b'IM01', b'IM03'),
 (b'IM01', b'IM04'),
 (b'IM02', b'IC01'),
 (b'IM02', b'IC02'),
 (b'IM02', b'IC03'),
 (b'IM02', b'IC04'),
 (b'IM02', b'IF01'),
 (b'IM02', b'IF02'),
 (b'IM02', b'IF04'),
 (b'IM02', b'IH02'),
 (b'IM02', b'IH03'),
 (b'IM02', b'IH04'),
 (b'IM02', b'IM01'),
 (b'IM02', b'IM03'),
 (b'IM02', b'IM04'),
 (b'IM03', b'IH04'),
 (b'IM03', b'IM04')]

def load_data(start_time, end_time, data_dir=DATA_DIR):

    data = []
    files = [i for i in os.listdir(data_dir) if i.endswith('pkl') and i.startswith('20') and i >= f'{start_time}.pkl' and i <= f'{end_time}.pkl']
    files.sort()

    for f in tqdm(files):

        path = os.path.join(data_dir, f)
        df = pd.read_pickle(path)
        futures_columns = [i for i in df[b'codename'].unique() if i.decode().startswith('I')]

        # 价量
        ret_df = get_ret_df(df, futures_columns)
        vol_df = get_vol_df(df, futures_columns)
        premium_df = get_premium_df(df, futures_columns)
        premium_diff_df = get_premium_diff_df(df, futures_columns)
        target_df = get_target_df(df, futures_columns)

        data.append({
            'ret_df': ret_df,
            'vol_df': vol_df,
            'premium_df': premium_df,
            'premium_diff_df': premium_diff_df,
            'target_df': target_df,
            'vol_mean': vol_df.mean(),
            'vol_std': vol_df.std(),
            'date' : f[:-4]
        })
    return data


def get_ret_df(df, futures_columns):
    return pd.pivot_table(df, columns=b'codename', index=b'time', values=b'ret')[futures_columns]



def get_vol_df(df, futures_columns):
    vol_df = pd.pivot_table(df, columns=b'codename', index=b'time', values=b'vol')[futures_columns]
    # 量标准化
    return vol_df

def get_premium_df(df, futures_columns):
    premium_df = pd.pivot_table(df, columns=b'codename', index=b'time', values=b'close')
    
    for i in [ihs, ics, ifs, ims]:
        columns, index = i
        if set(columns).issubset(premium_df.columns):
            premium_df[columns] = (premium_df[columns]  * 10000).divide(premium_df[index] , axis=0) - 10000
    premium_df = premium_df[futures_columns]
    
    return premium_df

def get_premium_diff_df(df, futures_columns):
    premium_diff_df = pd.pivot_table(df, columns=b'codename', index=b'time', values=b'ret')
    for i in [ihs, ics, ifs, ims]:
        columns, index = i
        if set(columns).issubset(premium_diff_df.columns):
            premium_diff_df[columns] = premium_diff_df[columns].subtract(premium_diff_df[index], axis=0) 
    premium_diff_df = premium_diff_df[futures_columns]
    return premium_diff_df

def get_target_df(df, futures_columns):
    close_df = pd.pivot_table(df, columns=b'codename', index=b'time', values=b'close')[futures_columns]
    target_df = (close_df.rolling(20*5).mean().shift(-20*5) * 10000).divide(close_df, axis=0)- 10000
    return target_df

def get_n_days(index, n, data, future_id):
    
    futures_types = [future_id.decode()[-2:]]
    if future_id.decode()[:-2] == 'IM' and data[index-n+1]['date'] < '20220722':
        return None
    
    delievery_happens = False
    for i in range(1, n):
        future_type = futures_types[0]
        date = data[index-i]['date']
        if date in DELIVERY_DATE:
            if date[4:6] in CHANGE_MONTH:
                if future_type == '01':
                    prev_future_type = '02'
                elif future_type == '02':
                    prev_future_type = '03'
                elif future_type == '03':
                    prev_future_type = '04'
                elif future_type == '04':
                    return None
                else:
                    raise Exception('error')
            else:
                if future_type == '01':
                    prev_future_type = '02'
                elif future_type == '02':
                    return None
                elif future_type == '03':
                    prev_future_type = '03'
                elif future_type == '04':
                    prev_future_type = '04'
                else:
                    raise Exception('error')
            futures_types.insert(0, prev_future_type)
        else:
            futures_types.insert(0, future_type)
    futures_types = [(future_id.decode()[:-2] + t).encode() for t in futures_types]
    # print(futures_types)
    days_data = pd.DataFrame(data[index-n+1:index+1]).drop('date', axis=1)
    for i in range(days_data.shape[0]):
        for j in range(days_data.shape[1]):
            if len(days_data.iloc[i, j][futures_types[i]].shape) == 0:
                days_data.iloc[i, j] = days_data.iloc[i, j][futures_types[i]].item()
            else:
                days_data.iloc[i, j] = days_data.iloc[i, j][futures_types[i]].values.copy()
    days_data['date'] = pd.DataFrame(data[index-n+1:index+1])['date']
    return days_data

def preprocess(df, ret_mean_std, premium_mean_std, premium_diff_mean_std):
    df = df.copy()
    df[['vol_mean', 'vol_std']] = df[['vol_mean', 'vol_std']].shift(1)
    df = df.iloc[1:]
    df['vol_df'] = (df['vol_df'] - df['vol_mean']) / df['vol_std']
    df['ret_df'] = (df['ret_df'] - ret_mean_std[0]) / ret_mean_std[1]
    df['premium_df'] = (df['premium_df'] - premium_mean_std[0]) / premium_mean_std[1]
    df['premium_diff_df'] = (df['premium_diff_df'] - premium_diff_mean_std[0]) / premium_diff_mean_std[1]
    return df


def get_next(data, window, need_preprocess, index=None, futures_chosen=None):
    if index is None:
        index = np.random.randint(window, len(data))
    futures = data[index-window]['ret_df'].columns
    data1 = None
    data2 = None
    if futures_chosen is None:
        while (data1 is None) or (data2 is None): 
            futures_chosen = np.random.choice(futures, 2, replace=False)
            data1 = get_n_days(index, window+1, data, futures_chosen[0])
            data2 = get_n_days(index, window+1, data, futures_chosen[1])
    else:
        data1 = get_n_days(index, window+1, data, futures_chosen[0])
        data2 = get_n_days(index, window+1, data, futures_chosen[1])   
        if data1 is None or data2 is None:
            return None
    if need_preprocess:
        data1 = preprocess(data1,RET_MEAN_STD, PREMIUM_MEAN_STD, PREMIUM_DIFF_MEAN_STD)
        data2 = preprocess(data2,RET_MEAN_STD, PREMIUM_MEAN_STD, PREMIUM_DIFF_MEAN_STD)
    return (to_numpy(data1, window), 
            to_numpy(data2, window), 
            np.stack(data1['target_df'].values).reshape(window, 240, -1)[:, :, -1][:, :-5], 
            np.stack(data2['target_df'].values).reshape(window, 240, -1)[:, :, -1][:, :-5],
            futures_chosen, 
            index)

def get_batch(data, window, batch_size):
    xs = []
    ys = []
    futures = []
    for i in range(batch_size):
        x1, x2, y1, y2, futures_chosen,  index   = get_next(data, window, True)
        xs.append(np.stack([x1, x2], axis=-1))
        ys.append(np.stack([y1, y2, y1-y2], axis=-1))
        futures.append([FUTURE_DICT[future.decode()[:2]] for future in futures_chosen])
    return np.stack(xs), np.stack(ys), np.stack(futures)


def to_numpy(data1, window):
    return np.stack([np.vstack(data1['ret_df'].values).reshape(window, 240, 20), 
    np.vstack(data1['vol_df'].values).reshape(window, 240, 20),
    np.vstack(data1['premium_df'].values).reshape(window, 240, 20),
    np.vstack(data1['premium_diff_df'].values).reshape(window, 240, 20)], axis=-1)



class FutureDataset(Dataset):
    def __init__(self, path, start_date, end_date):
        self.path = path
        self.files = [i for i in os.listdir(path) if i[:8] >=start_date and i[:8] <end_date]
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_name = self.files[idx]
        future1 = file_name[9:11]
        future2 = file_name[14:16]
        with open(f'{self.path}/{file_name}', 'rb') as f:
            X = np.load(f).astype(np.float32)
            y = np.load(f).astype(np.float32)
        futures = np.array([FUTURE_DICT[future1], FUTURE_DICT[future2]])
        return X, y, futures
