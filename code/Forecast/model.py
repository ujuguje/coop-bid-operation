"""
Forecast models for LMP price and solar generation.

EncoderDecoderLSTM:
  Encoder : LSTM(input=1, hidden=32, layers=1, dropout=0.2) — 24-step history
  Decoder : LSTM(input=3, hidden=32, layers=1, dropout=0.2) — 8-step horizon
  Head    : separate Linear(32→1) per step (8 independent heads)

VanillaTransformer (benchmark):
  Same input/output format as LSTM.
  Enc projection + TransformerEncoder(d=32, heads=4, layers=2, ffn=64)
  Dec projection + TransformerDecoder(same)
  Head: separate Linear(32→1) per step

Training config (both):
  Optimiser: Adam lr=1e-3 | Loss: MSELoss | Batch: 32
  Max epochs: 100 | Early stopping patience: 10
"""
import math
import torch
import torch.nn as nn


class EncoderDecoderLSTM(nn.Module):
    def __init__(self,
                 encoder_input_size: int = 1,
                 decoder_input_size: int = 3,
                 hidden_size:        int = 32,
                 num_layers:         int = 1,
                 output_size:        int = 1,
                 pred_length:        int = 8):
        super().__init__()
        self.encoder = nn.LSTM(encoder_input_size, hidden_size, num_layers,
                               dropout=0.2, batch_first=True)
        self.decoder = nn.LSTM(decoder_input_size, hidden_size, num_layers,
                               dropout=0.2, batch_first=True)
        # One linear head per forecast step (avoids parameter sharing across horizons)
        self.heads = nn.ModuleList([nn.Linear(hidden_size, output_size) for _ in range(pred_length)])
        self.pred_length = pred_length

    def forward(self, encoder_input, decoder_input, target=None):
        """
        Args:
            encoder_input : (B, seq_len, encoder_input_size)
            decoder_input : (B, pred_length, decoder_input_size)
            target        : optional, unused (kept for API compatibility with teacher-forcing callers)

        Returns:
            outputs : (B, pred_length, 1)
        """
        batch_size = decoder_input.size(0)

        _, (h, c) = self.encoder(encoder_input)
        h = h.view(self.encoder.num_layers, batch_size, -1)
        c = c.view(self.encoder.num_layers, batch_size, -1)

        outputs = torch.zeros(batch_size, self.pred_length, 1, device=decoder_input.device)
        for t in range(self.pred_length):
            dec_in = decoder_input[:, t, :].unsqueeze(1)
            out, (h, c) = self.decoder(dec_in, (h, c))
            outputs[:, t] = self.heads[t](out).squeeze(1)

        return outputs


# ---------------------------------------------------------------------------
# Vanilla Transformer (benchmark model)
# ---------------------------------------------------------------------------

class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 200, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class VanillaTransformer(nn.Module):
    """
    Encoder-Decoder Transformer for multi-step price/solar forecasting.
    Drop-in benchmark against EncoderDecoderLSTM: identical input/output API.

    Architecture:
      d_model=32, nhead=4, encoder_layers=2, decoder_layers=2, ffn=64, dropout=0.1
    """
    def __init__(self,
                 encoder_input_size: int = 1,
                 decoder_input_size: int = 3,
                 d_model:            int = 32,
                 nhead:              int = 4,
                 num_encoder_layers: int = 2,
                 num_decoder_layers: int = 2,
                 dim_feedforward:    int = 64,
                 dropout:            float = 0.1,
                 pred_length:        int = 8):
        super().__init__()
        self.enc_proj = nn.Linear(encoder_input_size, d_model)
        self.dec_proj = nn.Linear(decoder_input_size, d_model)
        self.pos_enc  = _PositionalEncoding(d_model, dropout=dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_encoder_layers)

        dec_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(dec_layer, num_decoder_layers)

        self.heads = nn.ModuleList([nn.Linear(d_model, 1) for _ in range(pred_length)])
        self.pred_length = pred_length

    def forward(self, encoder_input, decoder_input, target=None):
        """
        Args:
            encoder_input : (B, seq_len, encoder_input_size)
            decoder_input : (B, pred_length, decoder_input_size)
            target        : unused (API compatibility with train_model)
        Returns:
            outputs : (B, pred_length, 1)
        """
        enc = self.pos_enc(self.enc_proj(encoder_input))    # (B, 24, d_model)
        dec = self.pos_enc(self.dec_proj(decoder_input))    # (B, 8, d_model)
        memory  = self.transformer_encoder(enc)             # (B, 24, d_model)
        dec_out = self.transformer_decoder(dec, memory)     # (B, 8, d_model)
        outputs = torch.stack(
            [self.heads[t](dec_out[:, t, :]) for t in range(self.pred_length)], dim=1)
        return outputs  # (B, pred_length, 1)
