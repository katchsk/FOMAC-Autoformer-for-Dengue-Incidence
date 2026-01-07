# MarkovAutoformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp
from Embed import DataEmbedding_wo_pos


class MarkovTransitionModule(nn.Module):
    def __init__(self, d_model, n_states=32, dropout=0.1, tau=1.0, teacher_forcing_ratio=0.3):
        super(MarkovTransitionModule, self).__init__()
        self.d_model = d_model
        self.n_states = n_states
        self.tau = tau
        self.teacher_forcing_ratio = teacher_forcing_ratio

        # State encoder: maps features to state probabilities
        self.state_encoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_states)
        )

        # Transition matrix learner
        self.transition_learner = nn.Sequential(
            nn.Linear(n_states, n_states * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_states * 2, n_states * n_states)
        )

        # Output projection (state -> d_model)
        self.state_decoder = nn.Sequential(
            nn.Linear(n_states, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )

        self.dropout = nn.Dropout(dropout)
        self.last_transition_matrices = None
        self.last_state_probs = None  # Store for supervision loss

    def forward(self, x, gt_states=None):
        B, L, D = x.shape

        # Encode to state probabilities
        state_logits = self.state_encoder(x)  # (B, L, n_states)
        state_probs = F.softmax(state_logits / self.tau, dim=-1)  # (B, L, n)

        # TEACHER FORCING: Mix predicted states with ground truth during training
        if self.training and gt_states is not None:
            gt_one_hot = F.one_hot(gt_states, num_classes=self.n_states).float()  # (B, L, n)
            # Blend predicted and ground truth states
            state_probs = (1 - self.teacher_forcing_ratio) * state_probs + \
                         self.teacher_forcing_ratio * gt_one_hot

        # Store for supervision loss calculation
        self.last_state_probs = state_probs

        # Learn transition matrices for each timestep
        transition_matrices = []
        for t in range(L - 1):
            current_state = state_probs[:, t, :]  # (B, n)
            trans_logits = self.transition_learner(current_state)  # (B, n*n)
            trans_matrix = trans_logits.view(B, self.n_states, self.n_states)
            trans_matrix = F.softmax(trans_matrix / self.tau, dim=-1)  # row-wise prob
            transition_matrices.append(trans_matrix)

        if len(transition_matrices) == 0:
            transition_matrices = None
            self.last_transition_matrices = None
        else:
            transition_matrices = torch.stack(transition_matrices, dim=1)  # (B, T, n, n)
            
            # EMA smoothing
            if not hasattr(self, 'prev_trans_matrix') or self.prev_trans_matrix is None:
                self.prev_trans_matrix = transition_matrices.detach().clone()
            elif self.prev_trans_matrix.shape == transition_matrices.shape:
                transition_matrices = 0.9 * self.prev_trans_matrix.detach() + 0.1 * transition_matrices
                self.prev_trans_matrix = transition_matrices.detach().clone()
            else:
                self.prev_trans_matrix = transition_matrices.detach().clone()

            self.last_transition_matrices = transition_matrices

        # Decode state probabilities back to feature space
        markov_out = self.state_decoder(state_probs)  # (B, L, D)

        return markov_out, transition_matrices


class HybridEncoderLayer(nn.Module):
    def __init__(self, attention, d_model, n_states=32, d_ff=None,
                 moving_avg=25, dropout=0.1, activation="relu",
                 markov_weight=0.3, markov_tau=1.0, teacher_forcing_ratio=0.3):
        super(HybridEncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model

        self.attention = attention
        self.markov = MarkovTransitionModule(d_model, n_states, dropout, tau=markov_tau,
                                            teacher_forcing_ratio=teacher_forcing_ratio)
        self.markov_weight = markov_weight

        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, gt_states=None):
        attn_out, attn = self.attention(x, x, x, attn_mask=attn_mask)
        markov_out, trans_probs = self.markov(x, gt_states=gt_states)  # PASS gt_states
        # Weighted combination
        new_x = self.markov_weight * markov_out + (1 - self.markov_weight) * attn_out
        x = x + self.dropout(new_x)
        
        x, _ = self.decomp1(x)

        # FFN
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        res, _ = self.decomp2(x + y)

        return res, attn


class HybridDecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, c_out, n_states=32, d_ff=None,
                 moving_avg=25, dropout=0.1, activation="relu",
                 markov_weight=0.3, markov_tau=1.0, teacher_forcing_ratio=0.3):
        super(HybridDecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model

        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.markov = MarkovTransitionModule(d_model, n_states, dropout, tau=markov_tau,
                                            teacher_forcing_ratio=teacher_forcing_ratio)
        self.markov_weight = markov_weight
        self.c_out = c_out

        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        self.decomp3 = series_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

        self.projection = nn.Conv1d(
            in_channels=d_model,
            out_channels=self.c_out,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode='circular',
            bias=False
        )

    def forward(self, x, cross, x_mask=None, cross_mask=None, gt_states=None):
        attn_out = self.self_attention(x, x, x, attn_mask=x_mask)[0]
        markov_out, _ = self.markov(x, gt_states=gt_states)  # PASS gt_states

        combined = self.markov_weight * markov_out + (1 - self.markov_weight) * attn_out
        x = x + self.dropout(combined)
        x, trend1 = self.decomp1(x)

        x = x + self.dropout(self.cross_attention(x, cross, cross, attn_mask=cross_mask)[0])
        x, trend2 = self.decomp2(x)

        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x, trend3 = self.decomp3(x + y)

        residual_trend = trend1 + trend2 + trend3
        residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)

        return x, residual_trend


class MarkovAutoformer(nn.Module):
    def __init__(self, configs):
        super(MarkovAutoformer, self).__init__()
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.d_model = configs.d_model

        self.n_states = getattr(configs, 'n_states', 32)
        self.markov_weight = getattr(configs, 'markov_weight', 0.3)
        self.markov_tau = getattr(configs, 'markov_tau', 1.0)
        self.markov_supervised_weight = getattr(configs, 'markov_supervised_weight', 0.1)
        self.teacher_forcing_ratio = getattr(configs, 'teacher_forcing_ratio', 0.3)

        kernel_size = configs.moving_avg
        self.decomp = series_decomp(kernel_size)

        self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model,
                                                  configs.embed, configs.freq, configs.dropout)
        self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model,
                                                  configs.embed, configs.freq, configs.dropout)

        self.encoder = Encoder([
            HybridEncoderLayer(
                AutoCorrelationLayer(
                    AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                    configs.d_model,
                    configs.n_heads
                ),
                configs.d_model,
                self.n_states,
                configs.d_ff,
                moving_avg=configs.moving_avg,
                dropout=configs.dropout,
                activation=configs.activation,
                markov_weight=self.markov_weight,
                markov_tau=self.markov_tau,
                teacher_forcing_ratio=self.teacher_forcing_ratio
            )
            for _ in range(configs.e_layers)
        ], norm_layer=my_Layernorm(configs.d_model))

        self.decoder = Decoder(
            [
                HybridDecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(True, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads),
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.c_out,
                    self.n_states,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                    markov_weight=self.markov_weight,
                    markov_tau=self.markov_tau,
                    teacher_forcing_ratio=self.teacher_forcing_ratio
                )
                for _ in range(configs.d_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                seq_state_x=None, seq_state_y=None,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # Reset transition matrices
        for m in self.modules():
            if hasattr(m, 'last_transition_matrices'):
                m.last_transition_matrices = None
            if hasattr(m, 'last_state_probs'):
                m.last_state_probs = None

        # Decomposition
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)

        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)

        # Encoder with state supervision
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask, gt_states=seq_state_x)

        seq_state_y_aligned = None
        if seq_state_y is not None:
            dec_len = seasonal_init.shape[1]  # label_len + pred_len
            if seq_state_y.shape[1] >= dec_len:
                seq_state_y_aligned = seq_state_y[:, -dec_len:]
            elif seq_state_y.shape[1] < dec_len:
                pad_len = dec_len - seq_state_y.shape[1]
                last_state = seq_state_y[:, -1:].expand(-1, pad_len)
                seq_state_y_aligned = torch.cat([seq_state_y, last_state], dim=1)

        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        dec_out, trend = self.decoder(dec_out, enc_out, x_mask=dec_self_mask,
                                      cross_mask=dec_enc_mask, trend=trend_init, 
                                      gt_states=seq_state_y_aligned)
        
        dec_out = trend + dec_out

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], None
        else:
            return dec_out[:, -self.pred_len:, :]


class MarkovAutoformerConfig:
    def __init__(self):
        self.seq_len = 96
        self.label_len = 48
        self.pred_len = 24
        self.enc_in = 7
        self.dec_in = 7
        self.c_out = 7
        self.d_model = 512
        self.n_heads = 8
        self.e_layers = 2
        self.d_layers = 1
        self.d_ff = 2048
        self.moving_avg = 25
        self.factor = 3
        self.dropout = 0.05
        self.activation = 'gelu'
        self.output_attention = False
        self.embed = 'timeF'
        self.freq = 'h'
        
        self.n_states = 32
        self.markov_weight = 0.3
        self.markov_tau = 0.7
        self.markov_supervised_weight = 0.3
        self.teacher_forcing_ratio = 0.3 