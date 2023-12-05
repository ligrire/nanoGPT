import pandas as pd
import numpy as np
import torch

DAILY_MEAN = np.array([ 3.83512585e-04,
                        1.98364143e-02,
                        -1.91414416e-02,
                        -9.19866568e-04,
                        1.14106712e-02])
DAILY_STD = np.array([2.90859059e-02,
                        2.41179702e-02,
                        2.04035480e-02,
                        1.25962124e-02,
                        1.57830477e-01])

MINUTE_AMOUNT_MEAN = 827159.1461380336
MINUTE_AMOUNT_STD = 2997679.795808271

MINUTE_RATIO_MEAN = 4.461887552597751e-05
MINUTE_RATIO_STD = 0.023476901640949538

class MarketDataset(torch.utils.data.Dataset):
    def __init__(self, files, index_files, daily_mean=DAILY_MEAN, daily_std=DAILY_STD,  to_mean=MINUTE_AMOUNT_MEAN, to_std=MINUTE_AMOUNT_STD, 
                  need_track=False, max_sample_size=1000000, ratio_mean=MINUTE_RATIO_MEAN, ratio_std=MINUTE_RATIO_STD) -> None:
        self.data = np.zeros((max_sample_size, 3248), dtype=np.float32)
        self.index_data = []
        files.sort()
        index_files.sort()
        smallest_date = files[0][-12:-4]
        self.trade_days = []
        start = False
        for i in range(1, len(index_files)):
            if not start and index_files[i+61][-12:-4] == smallest_date:
                start = True
            if start:
                last_df = pd.read_pickle(index_files[i-1])
                df = pd.read_pickle(index_files[i])
                self.index_data.append(df / last_df.iloc[-1, :].values - 1)
                self.trade_days.append(int(index_files[i][-12:-4]))
        if need_track:
            df_index = []
        self.df_day = []
        mapping_dict = {
            (-10, 10): 0,
            (-20, 20): 1,
        }
        num_sample = 0
        for f in files:
            df = pd.read_pickle(f)
            df['meta', 'limit'] = df['meta', 'limit'].apply(lambda x: mapping_dict[x])
            na_rows = df.isna().any(axis=1)
            if na_rows.any():
                print(f'na rows: {f}')
            df = df[~df.isna().any(axis=1)]
            self.data[num_sample:num_sample + len(df)] = df.values.astype(np.float32)
            num_sample += len(df)
            if need_track:
                df_index.append(df.index.values)
            self.df_day = self.df_day + [int(f[-12:-4])] * len(df)
        if need_track:
            self.df_index = np.hstack(df_index)
        self.df_day = np.array(self.df_day)

        self.data = self.data[:num_sample]
        self.data.flags.writeable = False
        print(f'load {num_sample} samples')
        self.daily_mean = daily_mean
        self.daily_std = daily_std
        self.to_mean = to_mean
        self.to_std = to_std

        self.ratio_mean = ratio_mean
        self.ratio_std = ratio_std

    def get_item_with_info(self, idx):
        return {
            'data': self.__getitem__(idx),
            'day': self.df_day[idx],
            'stock': self.df_index[idx],
        }
    
    def __getitem__(self, idx):
        current_day_index = self.trade_days.index(self.df_day[idx])

        index_close = []
        index_minute = []
        # for i in range(current_day_index - 61, current_day_index):
        #     last_df = self.index_data[self.trade_days[i-1]]
        #     index_df = self.index_data[self.trade_days[i]]
        #     index_close.append(index_df.iloc[-1, :].values / last_df.iloc[-1, :].values - 1)

        index_close = np.stack(self.index_data[current_day_index-61:current_day_index])[:, -1, :]
        index_close = (index_close - self.daily_mean[0]) / self.daily_std[0]

        # for i in range(current_day_index - 5, current_day_index + 1):
        #     last_df = self.index_data[self.trade_days[i-1]]
        #     index_df = self.index_data[self.trade_days[i]]
        #     index_minute.append(index_df / last_df.iloc[-1, :].values - 1)

        index_minute = np.stack(self.index_data[current_day_index-5:current_day_index+1])
        index_minute = (index_minute - self.ratio_mean)/ self.ratio_std


        daily_data = self.data[idx, :61*6].reshape(61, 6).copy()
        daily_data[:, :5] = (daily_data[:, :5] - self.daily_mean) / self.daily_std
 
        daily_data[:, 5] = (daily_data[:, 5] - daily_data[:, 5].mean()) / daily_data[:, 5].std()

        daily_data = np.concatenate([daily_data, index_close], axis=1)
        
        

        minute_data = self.data[idx, 61*6:61*6+240*6*2].reshape(6, 2, 240)[::-1, :, :].transpose(0, 2, 1)

        minute_data = (minute_data - np.array([self.to_mean, self.ratio_mean])) / np.array([self.to_std, self.ratio_std])

        minute_data = np.concatenate([minute_data, index_minute ], axis=2)

        zt_label = self.data[idx, -2]

        zt_limit = self.data[idx, -1]

        # feature: (5*5), (241*2), (1), (241)
        # label: (241), (0 or 1)
        return (daily_data.astype(np.float32), minute_data.astype(np.float32), zt_limit.astype(np.int32)), (zt_label.astype(np.float32), )

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    import random
    random.seed(42)
    np.random.seed(42)
    import os
    path = '/Users/shitiancheng/quant/github/sample_data'
    files = os.listdir(path)
    files = [os.path.join(path, f) for f in files if f.endswith('.pkl')]
    index_files = os.listdir(path+'/index')
    index_files = [os.path.join(path + '/index', f) for f in index_files if f.endswith('.pkl')] 
    dataset = MarketDataset(files, index_files=index_files, max_sample_size=10000, daily_mean=DAILY_MEAN * 0, daily_std=DAILY_STD * 0+1,  to_mean=MINUTE_AMOUNT_MEAN * 0, to_std=MINUTE_AMOUNT_STD * 0 + 1, 
                  need_track=True, ratio_mean=MINUTE_RATIO_MEAN * 0, ratio_std=MINUTE_RATIO_STD * 0 + 1)
    import random
    for i in range(10):
        rand_i = random.randint(0, len(dataset))
        x = dataset[rand_i]

