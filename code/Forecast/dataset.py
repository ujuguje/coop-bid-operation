"""
PyTorch Dataset for electricity price / solar generation forecasting.

Sliding-window construction:
  - Encoder input : 24 consecutive scaled price values (6-hour history)
  - Decoder input : same 8 target slots from the previous 3 days (teacher-forcing context)
  - Target        : scaled price values for the next 8 steps (2-hour horizon)

Scaling: MinMaxScaler(0, 1) on price; StandardScaler on optional volume data.
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class PriceForecastDataset(Dataset):
    def __init__(self,
                 price_data,
                 time_index,
                 seq_length:    int = 24,
                 pred_length:   int = 8,
                 samples_per_day: int = 96,
                 volume_data=None):
        """
        Args:
            price_data      : 1-D array-like of raw prices
            time_index      : same-length array of timestamps (for output labelling)
            seq_length      : encoder input length (timesteps)
            pred_length     : forecast horizon (timesteps)
            samples_per_day : 96 for 15-min resolution
            volume_data     : optional 1-D array for additional feature (StandardScaler)
        """
        self.seq_length      = seq_length
        self.pred_length     = pred_length
        self.samples_per_day = samples_per_day
        self.time_index      = time_index

        self.price_scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = self.price_scaler.fit_transform(np.array(price_data).reshape(-1, 1))
        self.price_data = torch.FloatTensor(scaled)

        if volume_data is not None:
            vol_scaler = StandardScaler()
            scaled_vol = vol_scaler.fit_transform(np.array(volume_data).reshape(-1, 1))
            self.data = torch.cat([self.price_data, torch.FloatTensor(scaled_vol)], dim=1)
        else:
            self.data = self.price_data

    def __len__(self):
        return len(self.data) - self.pred_length - 3 * self.samples_per_day + 1

    def __getitem__(self, idx):
        adj = idx + 3 * self.samples_per_day - self.seq_length

        encoder_input = self.data[adj: adj + self.seq_length]

        # Decoder context: same target slots from previous 3 days
        s = adj - self.samples_per_day + self.seq_length
        decoder_input = torch.stack([
            self.price_data[s:      s      + self.pred_length],
            self.price_data[s - self.samples_per_day:      s - self.samples_per_day      + self.pred_length],
            self.price_data[s - 2 * self.samples_per_day:  s - 2 * self.samples_per_day  + self.pred_length],
        ], dim=1).squeeze()

        target_prices = self.price_data[adj + self.seq_length: adj + self.seq_length + self.pred_length]
        target_times  = self.time_index[adj + self.seq_length: adj + self.seq_length + self.pred_length]

        return encoder_input, decoder_input, {'prices': target_prices, 'times': target_times}

    def inverse_transform(self, scaled):
        return self.price_scaler.inverse_transform(np.array(scaled).reshape(-1, 1)).flatten()
