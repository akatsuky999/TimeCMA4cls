"""
Script to generate embeddings for time series classification datasets.

This script loads the classification data and generates GPT-2 based
text embeddings that can be used by the TimeCMA model for cross-modal
alignment.

Usage:
    python scripts/generate_embeddings.py --dataset PEMS-SF --split train
    python scripts/generate_embeddings.py --dataset PEMS-SF --split test
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import numpy as np
import h5py
from tqdm import tqdm

from datasets.data import data_factory


def parse_args():
    parser = argparse.ArgumentParser(description="Generate embeddings for classification dataset")
    parser.add_argument("--dataset", type=str, required=True, 
                        help="Name of the dataset (e.g., PEMS-SF, FaceDetection)")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Data directory (default: ./datasets/{dataset})")
    parser.add_argument("--save_dir", type=str, default="./Embeddings",
                        help="Directory to save embeddings")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test", "val"],
                        help="Data split to process")
    parser.add_argument("--model_name", type=str, default="gpt2",
                        help="LLM model name")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    parser.add_argument("--use_simple", action="store_true",
                        help="Use simplified (faster) embedding generation")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for processing")
    return parser.parse_args()


def generate_embeddings_for_dataset(
    dataset_name: str,
    data_dir: str,
    save_dir: str,
    split: str,
    model_name: str = "gpt2",
    device: str = "cuda",
    use_simple: bool = True,
    batch_size: int = 1
):
    """
    Generate embeddings for a classification dataset.
    """
    from transformers import GPT2Tokenizer, GPT2Model
    
    # Determine pattern based on split
    if split == "train":
        pattern = "TRAIN"
    elif split == "test":
        pattern = "TEST"
    else:
        pattern = split.upper()
    
    # Load dataset
    print(f"Loading {dataset_name} {split} data from {data_dir}...")
    config = {
        'task': 'classification',
        'subsample_factor': None
    }
    
    data_class = data_factory.get('tsra')
    if data_class is None:
        raise ValueError("Dataset class 'tsra' not found")
    
    my_data = data_class(data_dir, pattern=pattern, n_proc=1, config=config)
    
    # Get data info
    all_ids = my_data.all_IDs
    feature_df = my_data.feature_df
    max_seq_len = my_data.max_seq_len
    num_features = feature_df.shape[1]
    
    print(f"Dataset info:")
    print(f"  - Number of samples: {len(all_ids)}")
    print(f"  - Sequence length: {max_seq_len}")
    print(f"  - Number of features: {num_features}")
    
    # Initialize GPT-2
    print(f"Loading {model_name} model...")
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2Model.from_pretrained(model_name).to(device)
    model.eval()
    
    d_model = model.config.hidden_size  # 768 for GPT-2
    
    # Create save directory
    save_path = os.path.join(save_dir, dataset_name, split)
    os.makedirs(save_path, exist_ok=True)
    
    print(f"Generating embeddings to {save_path}...")
    
    with torch.no_grad():
        for idx, sample_id in enumerate(tqdm(all_ids, desc=f"Processing {split}")):
            # Get sample data
            sample_data = feature_df.loc[sample_id].values  # (seq_len, num_features)
            
            if use_simple:
                # Simple embedding: use statistical features
                embeddings = generate_simple_embedding(sample_data, model, tokenizer, device, d_model)
            else:
                # Full embedding: generate text prompt for each channel
                embeddings = generate_full_embedding(sample_data, model, tokenizer, device, d_model)
            
            # Save embedding: [d_model, num_features]
            file_path = os.path.join(save_path, f"{idx}.h5")
            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('embeddings', data=embeddings)
    
    print(f"Done! Saved {len(all_ids)} embeddings to {save_path}")


def generate_simple_embedding(sample_data, model, tokenizer, device, d_model):
    """
    Generate simple embedding based on statistical features.
    
    Args:
        sample_data: (seq_len, num_features) array
        model: GPT-2 model
        tokenizer: GPT-2 tokenizer
        device: torch device
        d_model: embedding dimension
        
    Returns:
        embeddings: (d_model, num_features) array
    """
    seq_len, num_features = sample_data.shape
    embeddings = np.zeros((d_model, num_features), dtype=np.float32)
    
    # Pre-compute template embeddings
    templates = [
        "mean",
        "standard deviation", 
        "minimum",
        "maximum",
        "trend",
        "range"
    ]
    
    template_embs = []
    for template in templates:
        tokens = tokenizer.encode(template, return_tensors="pt").to(device)
        emb = model(tokens).last_hidden_state[0, -1, :].cpu().numpy()
        template_embs.append(emb)
    template_embs = np.stack(template_embs, axis=0)  # (6, d_model)
    
    for j in range(num_features):
        values = sample_data[:, j]
        
        # Compute statistics
        stats = np.array([
            np.mean(values),
            np.std(values),
            np.min(values),
            np.max(values),
            np.sum(np.diff(values)),  # trend
            np.max(values) - np.min(values)  # range
        ])
        
        # Normalize
        stats = (stats - np.mean(stats)) / (np.std(stats) + 1e-8)
        
        # Weighted combination
        emb = np.dot(stats, template_embs)  # (d_model,)
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        
        embeddings[:, j] = emb
    
    return embeddings


def generate_full_embedding(sample_data, model, tokenizer, device, d_model):
    """
    Generate full embedding by creating text prompt for each channel.
    
    Args:
        sample_data: (seq_len, num_features) array
        model: GPT-2 model
        tokenizer: GPT-2 tokenizer
        device: torch device
        d_model: embedding dimension
        
    Returns:
        embeddings: (d_model, num_features) array
    """
    seq_len, num_features = sample_data.shape
    embeddings = np.zeros((d_model, num_features), dtype=np.float32)
    
    for j in range(num_features):
        values = sample_data[:, j]
        
        # Subsample for prompt
        if len(values) > 50:
            indices = np.linspace(0, len(values)-1, 50, dtype=int)
            values_sub = values[indices]
        else:
            values_sub = values
        
        # Create prompt
        values_str = ", ".join([f"{v:.2f}" for v in values_sub])
        stats = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'trend': np.sum(np.diff(values))
        }
        
        prompt = (
            f"Time series channel {j+1} with {seq_len} time steps. "
            f"Values: {values_str}. "
            f"Mean={stats['mean']:.2f}, std={stats['std']:.2f}, "
            f"min={stats['min']:.2f}, max={stats['max']:.2f}, trend={stats['trend']:.2f}."
        )
        
        # Tokenize and get embedding
        tokens = tokenizer.encode(prompt, return_tensors="pt", 
                                  max_length=512, truncation=True).to(device)
        emb = model(tokens).last_hidden_state[0, -1, :].cpu().numpy()
        
        embeddings[:, j] = emb
    
    return embeddings


if __name__ == "__main__":
    args = parse_args()
    
    # Default data directory
    if args.data_dir is None:
        args.data_dir = f"./datasets/{args.dataset}"
    
    generate_embeddings_for_dataset(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        split=args.split,
        model_name=args.model_name,
        device=args.device,
        use_simple=args.use_simple,
        batch_size=args.batch_size
    )

