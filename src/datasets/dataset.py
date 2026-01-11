import os
import numpy as np
from torch.utils.data import Dataset
import torch
import h5py


class ClassiregressionDatasetWithEmb(Dataset):
    """
    Dataset class for classification/regression with pre-computed embeddings.
    
    This class loads pre-computed embeddings (from GPT-2 or DeepSeek-OCR)
    alongside the time series data, for use with TimeCMA cross-modal model.
    
    Embedding 格式: 每个样本一个 .h5 文件 (0.h5, 1.h5, ...)
    """

    def __init__(self, data, indices, embedding_dir=None, dataset_name=None, split_name='train'):
        """
        Initialize dataset with embedding support.
        
        Args:
            data: Base data object (subclass of BaseData)
            indices: List of sample IDs/indices
            embedding_dir: Directory containing pre-computed embeddings
            dataset_name: Name of the dataset (used for embedding path)
            split_name: Name of the split (train/test)
        """
        super(ClassiregressionDatasetWithEmb, self).__init__()

        self.data = data
        self.IDs = indices
        self.feature_df = self.data.feature_df.loc[self.IDs]
        self.labels_df = self.data.labels_df.loc[self.IDs]
        
        # Embedding configuration
        self.embedding_dir = embedding_dir
        self.dataset_name = dataset_name
        self.split_name = split_name
        self.use_embeddings = embedding_dir is not None and dataset_name is not None
        
        if self.use_embeddings:
            self.emb_path = os.path.join(embedding_dir, dataset_name, split_name)
            # Check if embedding directory exists
            if not os.path.exists(self.emb_path):
                print(f"Warning: Embedding directory {self.emb_path} does not exist. Embeddings will be disabled.")
                self.use_embeddings = False
        
        # Create mapping from sample ID to embedding index
        self.id_to_emb_idx = {sample_id: idx for idx, sample_id in enumerate(self.IDs)}

    def __getitem__(self, ind):
        """
        Get a sample with its embedding.
        
        Args:
            ind: Integer index of sample in dataset
            
        Returns:
            X: (seq_length, feat_dim) tensor of time series
            y: (num_labels,) tensor of labels
            embedding: (d_model, feat_dim) tensor of embeddings (or None if not available)
            ID: ID of sample
        """
        sample_id = self.IDs[ind]
        X = self.feature_df.loc[sample_id].values  # (seq_length, feat_dim)
        y = self.labels_df.loc[sample_id].values   # (num_labels,)
        
        # Load embedding if available
        embedding = None
        if self.use_embeddings:
            emb_idx = self.id_to_emb_idx[sample_id]
            emb_file = os.path.join(self.emb_path, f"{emb_idx}.h5")
            
            if os.path.exists(emb_file):
                try:
                    with h5py.File(emb_file, 'r') as hf:
                        embedding = hf['embeddings'][:]  # (d_model, feat_dim)
                        embedding = torch.from_numpy(embedding).float()
                except Exception as e:
                    print(f"Warning: Failed to load embedding from {emb_file}: {e}")
                    embedding = None
        
        return torch.from_numpy(X), torch.from_numpy(y), embedding, sample_id

    def __len__(self):
        return len(self.IDs)


def collate_superv_with_emb(data, max_len=None, d_model=768):
    """
    Build mini-batch tensors from samples with embeddings.
    
    Args:
        data: List of (X, y, embedding, ID) tuples
            - X: torch tensor of shape (seq_length, feat_dim)
            - y: torch tensor of shape (num_labels,)
            - embedding: torch tensor of shape (d_model, feat_dim) or None
            - ID: sample ID
        max_len: Global fixed sequence length
        d_model: Embedding dimension (default: 768 for GPT-2)
        
    Returns:
        X: (batch_size, padded_length, feat_dim) tensor
        targets: (batch_size, num_labels) tensor
        embeddings: (batch_size, d_model, feat_dim) tensor (or None if no embeddings)
        padding_masks: (batch_size, padded_length) boolean tensor
        IDs: tuple of sample IDs
    """
    batch_size = len(data)
    features, labels, embeddings_list, IDs = zip(*data)
    
    # Check if any sample has embeddings
    has_embeddings = any(emb is not None for emb in embeddings_list)
    
    # Stack and pad features
    lengths = [X.shape[0] for X in features]
    if max_len is None:
        max_len = max(lengths)
    
    feat_dim = features[0].shape[-1]
    X = torch.zeros(batch_size, max_len, feat_dim)
    
    for i in range(batch_size):
        end = min(lengths[i], max_len)
        X[i, :end, :] = features[i][:end, :]
    
    # Stack labels
    targets = torch.stack(labels, dim=0)
    
    # Stack embeddings if available
    if has_embeddings:
        # Get embedding dimension from first non-None embedding
        emb_sample = next((e for e in embeddings_list if e is not None), None)
        if emb_sample is not None:
            d_model_actual = emb_sample.shape[0]
            embeddings = torch.zeros(batch_size, d_model_actual, feat_dim)
            
            for i, emb in enumerate(embeddings_list):
                if emb is not None:
                    # Handle potential shape mismatches
                    if emb.shape[1] == feat_dim:
                        embeddings[i] = emb
                    elif emb.shape[1] < feat_dim:
                        # Pad embedding features
                        embeddings[i, :, :emb.shape[1]] = emb
                    else:
                        # Truncate embedding features
                        embeddings[i] = emb[:, :feat_dim]
        else:
            embeddings = None
    else:
        embeddings = None
    
    # Create padding masks
    padding_masks = padding_mask(torch.tensor(lengths, dtype=torch.int16), max_len=max_len)
    
    return X, targets, embeddings, padding_masks, IDs


def padding_mask(lengths, max_len=None):
    """
    Used to mask padded positions: creates a (batch_size, max_len) boolean mask from a tensor of sequence lengths,
    where 1 means keep element at this position (time step)
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max_val()
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))
