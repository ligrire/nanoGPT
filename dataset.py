import pandas as pd
import numpy as np
import torch
from scipy.stats import rankdata
from tqdm import tqdm

DAILY_MEAN = np.array([1.55050660e-03,
                       2.08595957e-02,
                       -1.82031690e-02,
                       -1.03080110e-03,
                       2.86408834e+00])
DAILY_STD = np.array([[0.026074,
                       0.02130087,
                       0.01606067,
                       0.01244642,
                       4.13153832]])
MINUTE_RET_MEAN = 2.8424111888884474e-06
MINUTE_RET_STD = 0.0022672324784259196
MINUTE_TO_MEAN = 0.011813153671562965
MINUTE_TO_STD = 0.03699517976566984

class MarketDataset(torch.utils.data.Dataset):
    def __init__(self, files, daily_mean=DAILY_MEAN, daily_std=DAILY_STD, ret_mean=MINUTE_RET_MEAN, ret_std=MINUTE_RET_STD, to_mean=MINUTE_TO_MEAN, to_std=MINUTE_TO_STD, 
                 up_threshold=0.04, need_track=False, max_sample_size=1000000) -> None:
        self.data = np.zeros((max_sample_size, 750), dtype=np.float32)
        if need_track:
            df_index = []
            df_day = []
        mapping_dict = {
            (-5, 5): 0,
            (-10, 10): 1,
            (-20, 20): 2,
            (0, 0): 3
        }
        num_sample = 0
        for f in tqdm(files):
            df = pd.read_pickle(f)
            df['meta', 'limit'] = df['meta', 'limit'].apply(lambda x: mapping_dict[x])
            df = df[~df.isna().any(axis=1)]
            self.data[num_sample:num_sample + len(df)] = df.values.astype(np.float32)
            num_sample += len(df)
            if need_track:
                df_index.append(df.index.values)
                df_day = df_day + [int(f[-12:-4])] * len(df)
        if need_track:
            self.df_day = np.array(df_day)
            self.df_index = np.hstack(df_index)
        self.data = self.data[:num_sample]
        print(f'load {num_sample} samples')
        self.daily_mean = daily_mean
        self.daily_std = daily_std
        self.ret_mean = ret_mean
        self.ret_std = ret_std
        self.to_mean = to_mean
        self.to_std = to_std
        self.up_threshold = up_threshold

    def get_item_with_info(self, idx):
        return {
            'data': self.__getitem__(idx),
            'day': self.df_day[idx],
            'stock': self.df_index[idx],
        }
    
    def __getitem__(self, idx):
        daily_data = self.data[idx, :25].reshape(5, 5)
        daily_data = (daily_data - self.daily_mean) / self.daily_std
        minute_data = self.data[idx, 25:241 * 2+ 25].reshape(2, 241).T
        no_trade_index = (minute_data[:, 1] == 0).astype(int)
        minute_data = (minute_data - np.array([self.ret_mean, self.to_mean])) / np.array([self.ret_std, self.to_std])
        minute_label = (self.data[idx, 241 * 2+ 25: 241 * 2+ 25 + 241] > self.up_threshold).astype(int) 
        zt_label = self.data[idx, -2]

        zt_limit = self.data[idx, -1]

        # feature: (5*5), (241*2), (1), (241)
        # label: (241), (0 or 1)
        return (daily_data.astype(np.float32), minute_data.astype(np.float32), zt_limit.astype(np.int32), no_trade_index.astype(np.float32)), (minute_label.astype(np.float32), zt_label.astype(np.float32))

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    import os
    path = '/Users/shitiancheng/quant/github/sample_data'
    files = os.listdir(path)
    files = [os.path.join(path, f) for f in files if f.endswith('.pkl')]
    dataset = MarketDataset(files)
    import random
    for i in range(10):
        rand_i = random.randint(0, len(dataset))
        x = dataset.get_item_with_info(rand_i)

