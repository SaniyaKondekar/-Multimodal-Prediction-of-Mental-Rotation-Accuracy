import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
import os

# AdvancedTabularEmbeddingModel (unchanged)
class AdvancedTabularEmbeddingModel(nn.Module):
    def __init__(self, input_dim, embedding_dim=32, hidden_dims=[64, 32], dropout=0.5,
                 use_attention=True, use_residual=True):
        super(AdvancedTabularEmbeddingModel, self).__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.use_attention = use_attention
        self.use_residual = use_residual
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )
        if use_attention:
            self.feature_attention = nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.ReLU(),
                nn.Linear(input_dim // 2, input_dim),
                nn.Sigmoid()
            )
        self.hidden_layers = nn.ModuleList()
        prev_dim = hidden_dims[0]
        for hidden_dim in hidden_dims[1:]:
            layer = nn.Sequential(
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.hidden_layers.append(layer)
            prev_dim = hidden_dim
        self.embedding_layer = nn.Sequential(
            nn.Linear(prev_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout * 0.3),
            nn.Linear(embedding_dim, max(16, embedding_dim // 2)),
            nn.ReLU(),
            nn.BatchNorm1d(max(16, embedding_dim // 2)),
            nn.Dropout(dropout * 0.3),
            nn.Linear(max(16, embedding_dim // 2), 2)
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if module.out_features == 2:
                    nn.init.xavier_uniform_(module.weight, gain=1.0)
                else:
                    nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x, return_embeddings=False, return_attention=False):
        original_x = x
        if self.use_attention:
            attention_weights = self.feature_attention(x)
            x = x * attention_weights
        x = self.input_projection(x)
        for layer in self.hidden_layers:
            x = layer(x)
        embeddings = self.embedding_layer(x)
        if return_embeddings:
            if return_attention and self.use_attention:
                return embeddings, attention_weights
            return embeddings
        logits = self.classifier(embeddings)
        if return_attention and self.use_attention:
            return logits, embeddings, attention_weights
        return logits, embeddings

# Define model_dir
model_dir = 'models/fusion'

# Load modality_features
modality_features_path = os.path.join(model_dir, 'modality_features.pkl')
if not os.path.exists(modality_features_path):
    raise FileNotFoundError(f"modality_features.pkl not found in {model_dir}. Ensure it was saved and downloaded correctly.")
modality_features = joblib.load(modality_features_path)
print(f"Loaded modality_features.pkl from {modality_features_path}")

# Function to separate modalities (unchanged)
def separate_modalities(df, modality_features):
    modality_dfs = {}
    for modality, features in modality_features.items():
        if modality not in ['metadata', 'target'] and features:
            valid_features = [f for f in features if f in df.columns]
            if valid_features:
                modality_data = df[valid_features].copy()
                modality_dfs[modality] = modality_data
                print(f"Separated {modality} with {len(valid_features)} features")
            else:
                print(f"Warning: No valid features found for {modality}")
    return modality_dfs

# Helper to load tree model (updated for LightGBM)
# Helper to load tree model (updated for LightGBM)
def load_tree_model(key, path):
    try:
        if 'xgboost' in key:
            model = xgb.XGBClassifier()
            model.load_model(path)
            # Ensure model is fitted
            if not hasattr(model, '_Booster') or not model.get_booster():
                raise ValueError(f"{key} failed to load booster")
            return model
        elif 'lightgbm' in key:
            # Load LightGBM Booster
            booster = lgb.Booster(model_file=path)
            # Create dummy data to initialize classifier
            num_features = booster.num_feature()
            # Use a larger dummy dataset to satisfy LightGBM constraints
            dummy_X = np.random.randn(100, num_features)  # 100 samples with random data
            dummy_y = np.array([0, 1] * 50)  # Balanced binary classes
            # Initialize LGBMClassifier with booster
            model = lgb.LGBMClassifier(n_estimators=booster.num_trees(), min_data_in_leaf=1, min_data_in_bin=1)
            model.fit(dummy_X, dummy_y, init_model=booster)
            print(f"Initialized {key} with dummy fit")
            return model
        elif 'catboost' in key:
            model = CatBoostClassifier()
            model.load_model(path)
            if not model.is_fitted():
                raise ValueError(f"{key} is not fitted")
            return model
        else:
            model = joblib.load(path)
            return model
    except Exception as e:
        print(f"Failed to load {key}: {str(e)}")
        return None

# Prediction function (updated with meta-model prediction checks)
def predict_new_data(df_new, strategy='smote_best', model_dir=model_dir, device='cpu'):
    modality_dfs = separate_modalities(df_new, modality_features)
    modalities = ['eeg', 'eye', 'gsr', 'facial']

    modality_info = {
        'smote_best': {m: 'tree' for m in modalities},
        'vae_best': {m: 'tree' for m in modalities},
        'overall_best': {'eeg': 'tree', 'eye': 'tree', 'gsr': 'tree', 'facial': 'tree'}
    }.get(strategy, {m: 'tree' for m in modalities})

    embeddings = []
    for modality in modalities:
        if modality in modality_dfs:
            X_mod = modality_dfs[modality]
            emb_type = modality_info.get(modality, 'tree')

            try:
                if emb_type == 'tree':
                    scaler_path = os.path.join(model_dir, f'scaler_smote_tree_{modality}.pkl' if strategy in ['smote_best', 'overall_best'] else f'scaler_vae_tree_{modality}.pkl')
                    if not os.path.exists(scaler_path):
                        print(f"Scaler not found: {scaler_path}")
                        continue
                    scaler = joblib.load(scaler_path)
                    X_scaled = scaler.transform(X_mod)

                    train_pred = []
                    for model_type in ['xgboost', 'lightgbm', 'catboost']:
                        key = f'{modality}_{model_type}'
                        ext = '.json' if 'xgboost' in model_type else '.txt' if 'lightgbm' in model_type else '.cbm'
                        path = os.path.join(model_dir, f'{key}{ext}')
                        if os.path.exists(path):
                            model = load_tree_model(key, path)
                            if model is None:
                                print(f"Skipping {key} due to loading failure")
                                continue
                            # Verify model is usable
                            try:
                                test_input = X_scaled[:1]
                                model.predict_proba(test_input)
                                pred_proba = model.predict_proba(X_scaled)
                                train_pred.append(pred_proba)
                            except Exception as e:
                                print(f"Failed to predict with {key}: {str(e)}")
                                continue

                    if train_pred:
                        mod_emb = np.hstack(train_pred)
                        embeddings.append(mod_emb)
                        print(f"Generated tree embeddings for {modality}: {mod_emb.shape}")
                    else:
                        print(f"No tree models loaded for {modality}")
                else:
                    scaler_path = os.path.join(model_dir, f'scaler_smote_neural_{modality}.pkl' if strategy in ['smote_best', 'overall_best'] else f'scaler_vae_neural_{modality}.pkl')
                    if not os.path.exists(scaler_path):
                        print(f"Scaler not found: {scaler_path}")
                        continue
                    scaler = joblib.load(scaler_path)
                    X_scaled = scaler.transform(X_mod)

                    model = AdvancedTabularEmbeddingModel(
                        input_dim=X_scaled.shape[1],
                        embedding_dim=32,
                        hidden_dims=[64, 32],
                        dropout=0.5
                    ).to(device)
                    model_path = os.path.join(model_dir, f'neural_smote_{modality}.pth' if strategy in ['smote_best', 'overall_best'] else f'neural_vae_{modality}.pth')
                    if not os.path.exists(model_path):
                        print(f"Neural model not found: {model_path}")
                        continue
                    model.load_state_dict(torch.load(model_path, map_location=device))
                    model.eval()

                    X_tensor = torch.FloatTensor(X_scaled).to(device)
                    with torch.no_grad():
                        mod_emb = model(X_tensor, return_embeddings=True).cpu().numpy()
                    embeddings.append(mod_emb)
                    print(f"Generated neural embeddings for {modality}: {mod_emb.shape}")
            except Exception as e:
                print(f"Failed to process {modality}: {str(e)}")
                continue

    if not embeddings:
        print("No embeddings generated for any modality")
        return None, None
    min_samples = min(emb.shape[0] for emb in embeddings)
    embeddings = [emb[:min_samples] for emb in embeddings]
    fused_emb = np.hstack(embeddings)
    print(f"Fused embeddings shape: {fused_emb.shape}")

    meta_scaler_path = os.path.join(model_dir, f'meta_scaler_{strategy}.pkl')
    if not os.path.exists(meta_scaler_path):
        print(f"Meta scaler not found: {meta_scaler_path}")
        return None, None
    meta_scaler = joblib.load(meta_scaler_path)
    fused_scaled = meta_scaler.transform(fused_emb)

    meta_selector_path = os.path.join(model_dir, f'meta_selector_{strategy}.pkl')
    if os.path.exists(meta_selector_path):
        try:
            meta_selector = joblib.load(meta_selector_path)
            fused_scaled = meta_selector.transform(fused_scaled)
            print(f"Applied feature selection: {fused_scaled.shape}")
        except Exception as e:
            print(f"Feature selection failed: {str(e)}")

    meta_models = {}
    for name in ['logistic', 'random_forest', 'xgboost', 'lightgbm']:
        key = f'meta_{strategy}_{name}'
        ext = '.pkl' if name in ['logistic', 'random_forest'] else '.json' if name == 'xgboost' else '.txt'
        path = os.path.join(model_dir, f'{key}{ext}')
        if os.path.exists(path):
            try:
                if name == 'logistic':
                    meta_models[name] = joblib.load(path)
                elif name == 'random_forest':
                    meta_models[name] = joblib.load(path)
                elif name == 'xgboost':
                    meta_models[name] = xgb.XGBClassifier()
                    meta_models[name].load_model(path)
                elif name == 'lightgbm':
                    booster = lgb.Booster(model_file=path)
                    meta_models[name] = lgb.LGBMClassifier(n_estimators=booster.num_trees(), min_data_in_leaf=1, min_data_in_bin=1)
                    meta_models[name]._Booster = booster
                    # Perform dummy fit with larger dataset
                    num_features = booster.num_feature()
                    dummy_X = np.random.randn(100, num_features)  # 100 samples
                    dummy_y = np.array([0, 1] * 50)  # Balanced binary classes
                    meta_models[name].fit(dummy_X, dummy_y, init_model=booster)
                    print(f"Initialized meta-model {name} with dummy fit")
                # Verify meta-model is usable
                try:
                    test_input = fused_scaled[:1]
                    meta_models[name].predict_proba(test_input)
                    print(f"Verified meta-model {name} is ready for prediction")
                except Exception as e:
                    print(f"Meta-model {name} not ready for prediction: {str(e)}")
                    meta_models.pop(name, None)
                    continue
                print(f"Loaded meta-model: {name}")
            except Exception as e:
                print(f"Failed to load meta-model {name}: {str(e)}")

    if not meta_models:
        print("No meta-models loaded")
        return None, None

    probabilities = []
    for name, model in meta_models.items():
        try:
            prob = model.predict_proba(fused_scaled)[:, 1]
            probabilities.append(prob)
            print(f"Generated predictions from {name}")
        except Exception as e:
            print(f"Prediction failed for {name}: {str(e)}")

    if not probabilities:
        print("No predictions generated")
        return None, None

    ensemble_prob = np.mean(probabilities, axis=0)
    ensemble_pred = (ensemble_prob > 0.5).astype(int)
    print(f"Generated ensemble predictions: {ensemble_pred.shape}")

    return ensemble_pred, ensemble_prob



# Example usage
df_new = pd.read_csv('data.csv')
pred, prob = predict_new_data(df_new, strategy='smote_best', model_dir=model_dir)
if pred is not None:
    print("Predictions:", pred)
    print("Probabilities:", prob)
else:
    print("Prediction failed")
