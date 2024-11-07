import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm



class Dataset_ETT_hour(Dataset):
    def __init__(self, df, flag='train', size=None,
                 features='S', target='OT', scale=True, timeenc=0, freq='h', percent=100):
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        # Initialize settings
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.df = df
        self.percent = percent
        self.features = 'M'
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        # Load and process data
        self.__read_data__()

        # Set the encoding input size based on the number of features
        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - (self.seq_len + self.pred_len) + 1

    def __read_data__(self):
        df_raw = self.df

        # drop colum 'ID'
        df_raw.drop(['ID'], axis=1, inplace=True)

        hours_in_year = 24 * 30 * 12
        hours_in_four_months = 24 * 30 * 4

        train_time_step_start, train_time_step_end = 0, hours_in_year
        val_time_step_start, val_time_step_end = train_time_step_end - self.seq_len, train_time_step_end + hours_in_four_months
        test_time_step_start, test_time_step_end = val_time_step_end - self.seq_len, val_time_step_end + hours_in_four_months

        border1s = [train_time_step_start, val_time_step_start, test_time_step_start]
        border2s = [train_time_step_end, val_time_step_end, test_time_step_end]

        # Based on the learning mode select slice index
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        if self.features == 'M':
            df_data = df_raw[df_raw.columns[1:]]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        # TODO: Implement scaling
        #data = self.scaler.transform(df_data.values)

        data = df_data.values

        df_stamp = df_raw[['Date']][border1:border2]
        df_stamp['Date'] = pd.to_datetime(df_stamp.Date)
        df_stamp['month'] = df_stamp.Date.dt.month
        df_stamp['day'] = df_stamp.Date.dt.day
        df_stamp['weekday'] = df_stamp.Date.dt.weekday
        df_stamp['hour'] = df_stamp.Date.dt.hour
        data_stamp = df_stamp.drop(['Date'], axis=1).values

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len
        s_end = s_begin + self.seq_len

        #r_begin = s_end - self.label_len
        #r_end = r_begin + self.label_len + self.pred_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id + 1]

        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        # Convert sequences to tensors
        seq_x = torch.tensor(seq_x, dtype=torch.float32)
        seq_y = torch.tensor(seq_y, dtype=torch.float32)
        seq_x_mark = torch.tensor(seq_x_mark, dtype=torch.float32)
        seq_y_mark = torch.tensor(seq_y_mark, dtype=torch.float32)

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return self.tot_len


def main():
    df_raw = pd.read_csv("ETTh1.csv")
    print(df_raw.head())

    """
    seq_len 512
    lable_len 48
    pred_len 96
    factor 3
    enc in 7
    dec in 7
    c_out 7
    batch size 24
    d_model=32
    d_ff=128
    """
    dataset = Dataset_ETT_hour()

    data_loader = DataLoader(
        dataset,
        batch_size=24,
        shuffle=False,
        num_workers=1,
        drop_last=True
    )

    # i, (seqs, labels, seq_m, label_m)
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(data_loader)):
        print(f"batch {i + 1}:")
        print("\nbatch x: This is the input sequence for the models, containing the historical time series data points that the models will use to predict future values.")
        print("batch x:", batch_x.shape)
        print("\nbatch y: This is the target sequence for the models, containing the true future values that the models will try to predict.")
        print("batch y:", batch_y.shape)
        print("\nbatch x': This is the time encoding for the input sequence, which provides additional information about the time of day, day of the week, etc.")
        print("batch x':", batch_x_mark.shape)
        print("\nbatch y': This is the time encoding for the target sequence, which provides additional information about the time of day, day of the week, etc.")
        print("batch y':", batch_y_mark.shape)
        print("\n")

        if i == 2:
            break


if __name__ == "__main__":
    main()