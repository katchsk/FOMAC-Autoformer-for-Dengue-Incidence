# markov_autoformer_pipeline.py
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import joblib
import warnings
warnings.filterwarnings('ignore')


class DengueMarkovDataset(Dataset):
    def __init__(self, data_df, seq_len=156, label_len=52, pred_len=52,
                 features='MS', target='cases_minmax', scale=True, scaler=None,
                 stride=1, is_training=True, debug=False):
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.features = features
        self.target = target
        self.debug = debug
        self.stride = stride
        self.is_training = is_training

        self.df = data_df.copy().reset_index(drop=True)

        # Feature columns (exclude meta and state)
        self.feature_cols = [col for col in self.df.columns if col not in ['date', 'location', 'cases', 'state']]
        if target not in self.feature_cols:
            raise ValueError(f"Target '{target}' not found in features")
        self.target_idx = self.feature_cols.index(target)

        # Data and observed states
        self.data = self.df[self.feature_cols].values
        self.states = self.df['state'].astype(int).values

        # Scaling
        self.scaler = scaler
        if scale:
            if self.scaler is None:
                from sklearn.preprocessing import StandardScaler
                self.scaler = StandardScaler(with_mean=True, with_std=True)
                self.data = self.scaler.fit_transform(self.data)
            else:
                self.data = self.scaler.transform(self.data)

        self.data_stamp = self._get_time_features()

    def _get_time_features(self):
        time_features = []
        if 'month_sin' in self.df.columns and 'month_cos' in self.df.columns:
            time_features.append(self.df['month_sin'].values)
            time_features.append(self.df['month_cos'].values)
        if 'week_of_year_sin' in self.df.columns and 'week_of_year_cos' in self.df.columns:
            time_features.append(self.df['week_of_year_sin'].values)
            time_features.append(self.df['week_of_year_cos'].values)
        if len(time_features) > 0:
            return np.stack(time_features, axis=-1)
        else:
            return np.arange(len(self.data))[:, None] / len(self.data)

    def __len__(self):
        return max(0, (len(self.data) - self.seq_len - self.pred_len + 1) // self.stride)

    def __getitem__(self, index):
        s_begin = index * self.stride
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        if r_end > len(self.data):
            r_end = len(self.data)
            r_begin = r_end - (self.label_len + self.pred_len)
            s_end = r_begin + self.label_len
            s_begin = s_end - self.seq_len

        seq_x = self.data[s_begin:s_end].copy()  # Copy to avoid in-place modification
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y = self.data[r_begin:r_end].copy()
        seq_y_mark = self.data_stamp[r_begin:r_end]

        seq_state_x = self.states[s_begin:s_end].astype(int)
        seq_state_y = self.states[r_begin:r_end].astype(int)

        # Ensure lengths match
        min_len_x = min(len(seq_x), len(seq_x_mark))
        seq_x = seq_x[:min_len_x]
        seq_x_mark = seq_x_mark[:min_len_x]
        seq_state_x = seq_state_x[:min_len_x]

        min_len_y = min(len(seq_y), len(seq_y_mark))
        seq_y = seq_y[:min_len_y]
        seq_y_mark = seq_y_mark[:min_len_y]
        seq_state_y = seq_state_y[:min_len_y]

        if self.features == 'MS':
            seq_y_target = self.data[r_begin:r_end, self.target_idx:self.target_idx+1][:min_len_y]
        else:
            seq_y_target = self.data[r_begin:r_end][:min_len_y]

        if self.is_training:
            noise_std = 0.01
            seq_x = seq_x + np.random.normal(0, noise_std, seq_x.shape)
            seq_y = seq_y + np.random.normal(0, noise_std, seq_y.shape)

        return (torch.FloatTensor(seq_x),
                torch.FloatTensor(seq_x_mark),
                torch.FloatTensor(seq_y),
                torch.FloatTensor(seq_y_mark),
                torch.FloatTensor(seq_y_target),
                torch.LongTensor(seq_state_x),
                torch.LongTensor(seq_state_y))


class MarkovAutoformerTrainer:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction='mean')

    def _collect_state_probs_and_targets(self, seq_state_x):
        loss_terms = []
        for m in self.model.modules():
            if hasattr(m, 'last_state_probs') and m.last_state_probs is not None:
                state_probs = m.last_state_probs  # (B, L, n_states)
                B, L, n_states = state_probs.shape
                
                # Ensure target length matches
                target_len = min(L, seq_state_x.shape[1])
                probs = state_probs[:, :target_len, :]
                targets = seq_state_x[:, :target_len]
                
                loss_terms.append((probs, targets))
        
        return loss_terms

    def _compute_state_supervision_loss(self, seq_state_x):
        loss_terms = self._collect_state_probs_and_targets(seq_state_x)
        
        if len(loss_terms) == 0:
            return torch.tensor(0., device=self.device)
        
        total_loss = 0.0
        for state_probs, targets in loss_terms:
            # state_probs: (B, L, n_states)
            # targets: (B, L)
            
            # Reshape for cross-entropy
            probs_flat = state_probs.reshape(-1, state_probs.shape[-1])  # (B*L, n_states)
            targets_flat = targets.reshape(-1)  # (B*L,)
            
            # Cross-entropy loss
            loss = self.ce_loss(probs_flat, targets_flat)
            total_loss += loss
        
        return total_loss / len(loss_terms)

    def train_epoch(self, train_loader, optimizer, criterion):
        self.model.train()
        total_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            seq_x, seq_x_mark, seq_y, seq_y_mark, seq_y_target, seq_state_x, seq_state_y = batch

            seq_x = seq_x.to(self.device)
            seq_x_mark = seq_x_mark.to(self.device)
            seq_y = seq_y.to(self.device)
            seq_y_mark = seq_y_mark.to(self.device)
            seq_y_target = seq_y_target.to(self.device)
            seq_state_x = seq_state_x.to(self.device)
            seq_state_y = seq_state_y.to(self.device)

            optimizer.zero_grad()
            
            # Pass states to model
            outputs = self.model(seq_x, seq_x_mark, seq_y, seq_y_mark,
                               seq_state_x=seq_state_x, seq_state_y=seq_state_y)

            # Keep only dengue channel
            if outputs.shape[-1] > 1:
                outputs = outputs[..., 0:1]

            pred = outputs if not isinstance(outputs, (tuple, list)) else outputs[0]

            # Main forecasting loss
            pred_len = min(pred.shape[1], seq_y_target.shape[1])
            pred = pred[:, -pred_len:, :]
            target = seq_y_target[:, -pred_len:, :]
            main_loss = criterion(pred, target)

            # State supervision loss
            state_loss = self._compute_state_supervision_loss(seq_state_x)

            # Total loss
            weight = getattr(self.model, 'markov_supervised_weight', 0.3)
            loss = main_loss + weight * state_loss

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        return total_loss / max(1, len(train_loader))

    def validate(self, val_loader, criterion):
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                seq_x, seq_x_mark, seq_y, seq_y_mark, seq_y_target, seq_state_x, seq_state_y = batch

                seq_x = seq_x.to(self.device)
                seq_x_mark = seq_x_mark.to(self.device)
                seq_y = seq_y.to(self.device)
                seq_y_mark = seq_y_mark.to(self.device)
                seq_y_target = seq_y_target.to(self.device)
                seq_state_x = seq_state_x.to(self.device)
                seq_state_y = seq_state_y.to(self.device)

                # Pass states to model (but no teacher forcing in eval mode)
                outputs = self.model(seq_x, seq_x_mark, seq_y, seq_y_mark,
                                   seq_state_x=seq_state_x, seq_state_y=seq_state_y)
                
                if outputs.shape[-1] > 1:
                    outputs = outputs[..., 0:1]

                pred = outputs if not isinstance(outputs, (tuple, list)) else outputs[0]
                pred_len = min(pred.shape[1], seq_y_target.shape[1])
                pred = pred[:, -pred_len:, :]
                target = seq_y_target[:, -pred_len:, :]
                main_loss = criterion(pred, target)

                state_loss = self._compute_state_supervision_loss(seq_state_x)
                weight = getattr(self.model, 'markov_supervised_weight', 0.3)
                loss = main_loss + weight * state_loss

                total_loss += loss.item()
                num_batches += 1

        return total_loss / max(1, num_batches)

    def predict(self, test_loader):
        self.model.eval()
        predictions = []
        targets = []

        with torch.no_grad():
            for batch in test_loader:
                seq_x, seq_x_mark, seq_y, seq_y_mark, seq_y_target, seq_state_x, seq_state_y = batch

                seq_x = seq_x.to(self.device)
                seq_x_mark = seq_x_mark.to(self.device)
                seq_y = seq_y.to(self.device)
                seq_y_mark = seq_y_mark.to(self.device)
                seq_y_target = seq_y_target.to(self.device)
                seq_state_x = seq_state_x.to(self.device)
                seq_state_y = seq_state_y.to(self.device)

                # Pass states to model
                outputs = self.model(seq_x, seq_x_mark, seq_y, seq_y_mark,
                                   seq_state_x=seq_state_x, seq_state_y=seq_state_y)
                
                pred = outputs[0] if isinstance(outputs, (tuple, list)) else outputs

                if pred.shape[-1] > 1:
                    pred = pred[..., 0:1]

                pred_len = min(pred.shape[1], seq_y_target.shape[1])
                pred = pred[:, -pred_len:, :].cpu().numpy()
                target = seq_y_target[:, -pred_len:, :].cpu().numpy()

                predictions.append(pred)
                targets.append(target)

        predictions = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0)
        return predictions, targets


def prepare_data(features_path, seq_len=156, label_len=52, pred_len=52,
                 train_ratio=0.8, val_ratio=0.1, batch_size=16, stride=1, location=None):
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler, LabelEncoder
    from torch.utils.data import DataLoader
    import joblib

    # Load features
    features_df = pd.read_csv(features_path)

    # Filter by location if specified
    if location:
        features_df = features_df[features_df['location'] == location].reset_index(drop=True)

    # Sort by date
    features_df['date'] = pd.to_datetime(features_df['date'])
    features_df = features_df.sort_values('date').reset_index(drop=True)

    # Add lag features
    if all(col in features_df.columns for col in ['RAINFALL_minmax', 'TMAX_minmax', 'RH_minmax']):
        for lag in [6, 8, 12]:
            features_df[f'RAIN_L{lag}'] = features_df['RAINFALL_minmax'].shift(lag)
            features_df[f'TMAX_L{lag}'] = features_df['TMAX_minmax'].shift(lag)
            features_df[f'RH_L{lag}'] = features_df['RH_minmax'].shift(lag)
        features_df = features_df.dropna().reset_index(drop=True)
        print(f"✓ Added lagged weather features (lags: 6, 8, 12)")

    # Select relevant features
    selected_features = [
        'date', 'location',
        'cases_minmax', 'RAINFALL_minmax', 'TMAX_minmax', 'TMIN_minmax', 'RH_minmax',
        'RAIN_L6', 'RAIN_L8', 'RAIN_L12',
        'TMAX_L6', 'TMAX_L8', 'TMAX_L12',
        'RH_L6', 'RH_L8', 'RH_L12',
        'WIND_SPEED_minmax', 'WIND_DIR_X', 'WIND_DIR_Y',
        'month_sin', 'month_cos', 'week_of_year_sin', 'week_of_year_cos',
        'RAINFALL_minmax_roll_mean_4w', 'RAINFALL_minmax_lag_1w', 'RAINFALL_minmax_lag_2w', 'RAINFALL_minmax_lag_3w',
        'TMAX_minmax_roll_mean_4w', 'TMAX_minmax_lag_1w',
        'TMIN_minmax_roll_mean_4w', 'TMIN_minmax_lag_1w',
        'RH_minmax_roll_mean_4w', 'RH_minmax_lag_1w',
        'cases_minmax_roll_mean_4w', 'cases_minmax_lag_1w',
        'cases_minmax_lag_2w', 'cases_minmax_lag_3w', 'cases_minmax_lag_4w',
        'TMAX_x_RH', 'state'
    ]

    features_df = features_df[[col for col in selected_features if col in features_df.columns]]

    # Split into Train / Validation / Test
    n = len(features_df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    overlap = seq_len + pred_len

    val_start = max(0, train_end - overlap)
    test_start = max(0, val_end - overlap)

    train_df = features_df.iloc[:train_end].copy()
    val_df = features_df.iloc[val_start:val_end].copy()
    test_df = features_df.iloc[test_start:].copy()

    # Ensure 'state' is integer label
    for df in [train_df, val_df, test_df]:
        if 'state' in df.columns:
            if df['state'].dtype == object:
                le = LabelEncoder()
                df['state'] = le.fit_transform(df['state'].astype(str))
            df['state'] = df['state'].astype(int)

    feature_cols = [c for c in features_df.columns if c not in ['date', 'location']]
    cat_cols = [c for c in feature_cols if train_df[c].dtype == 'object' and c != 'state']
    num_cols = [c for c in feature_cols if c not in cat_cols and c != 'state']

    # Encode categorical features
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col].astype(str))
        val_df[col] = le.transform(val_df[col].astype(str))
        test_df[col] = le.transform(test_df[col].astype(str))
        label_encoders[col] = le

    # Scale numeric features (skip cases_minmax properly)
    scalers_dict = {}
    already_scaled = ['cases_minmax', 'cases_minmax_roll_mean_4w', 'cases_minmax_lag_1w',
                     'cases_minmax_lag_2w', 'cases_minmax_lag_3w', 'cases_minmax_lag_4w']
    
    for col in num_cols:
        if col in already_scaled:
            print(f"Skipping scaling for {col} (already scaled 0-1)")
            continue

        scaler = MinMaxScaler(feature_range=(0, 1))
        train_df[col] = scaler.fit_transform(train_df[[col]])
        val_df[col] = scaler.transform(val_df[[col]])
        test_df[col] = scaler.transform(test_df[[col]])
        scalers_dict[col] = scaler

    joblib.dump(scalers_dict, 'scalers_dict.pkl')
    print("Saved scalers to scalers_dict.pkl")

    # Create datasets with is_training flag
    train_dataset = DengueMarkovDataset(train_df, seq_len=seq_len, label_len=label_len,
                                       pred_len=pred_len, scale=False, stride=stride, is_training=True)
    val_dataset = DengueMarkovDataset(val_df, seq_len=seq_len, label_len=label_len,
                                     pred_len=pred_len, scale=False, stride=stride, is_training=False)
    test_dataset = DengueMarkovDataset(test_df, seq_len=seq_len, label_len=label_len,
                                      pred_len=pred_len, scale=False, stride=stride, is_training=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, scalers_dict, feature_cols


def calculate_metrics(predictions, targets):
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    pred_flat = predictions.reshape(-1)
    target_flat = targets.reshape(-1)

    mse = mean_squared_error(target_flat, pred_flat)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(target_flat, pred_flat)
    r2 = r2_score(target_flat, pred_flat)
    smape = 100 * np.mean(2 * np.abs(pred_flat - target_flat) / (np.abs(pred_flat) + np.abs(target_flat) + 1e-8))

    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'SMAPE': smape
    }