"""
Utility helpers for generating embeddings for time series classification datasets.

Following TimeCMA's design:
- For each sample (T, D), generate embedding for each of the D dimensions
- Each dimension's time series -> Prompt -> GPT2 -> (E, 1)
- Concatenate all D dimensions -> (E, D)

Usage:
    from emb_utils import generate_dataset_embeddings
    
    generate_dataset_embeddings(
        dataset_name="PEMS-SF",
        use_simple=False
    )
"""
from __future__ import annotations

import os
import sys
import re
import glob
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Add storage to path for importing GenClassificationEmb
sys.path.insert(0, str(PROJECT_ROOT / "storage"))


def load_ts_classification_data(data_dir: str, split: str = "train"):
    """
    Load .ts format classification dataset (without sktime dependency).
    
    Args:
        data_dir: Dataset directory (contains *_TRAIN.ts and *_TEST.ts files)
        split: "train" or "test"
    
    Returns:
        data_array: numpy array of shape (num_samples, seq_len, num_features)
        labels: numpy array of labels
        max_seq_len: maximum sequence length
        num_features: number of features
    """
    name = Path(data_dir).name
    tag = 'TRAIN' if split == 'train' else 'TEST'
    ts_path = Path(data_dir) / f"{name}_{tag}.ts"
    
    if not ts_path.exists():
        raise FileNotFoundError(f"File not found: {ts_path}")
    
    print(f"  Loading: {ts_path}")
    
    # Read file
    lines = ts_path.read_text(encoding='utf-8').splitlines()
    
    # Find @data line
    data_start = None
    for i, line in enumerate(lines):
        if line.strip().lower().startswith('@data'):
            data_start = i + 1
            break
    
    if data_start is None:
        raise ValueError(f"@data not found in {ts_path}")
    
    # Parse data lines
    raw_lines = [l.strip() for l in lines[data_start:] if l.strip()]
    num_samples = len(raw_lines)
    
    # Parse first line to get number of features
    # NOTE: .ts format is "dim1:dim2:...:dimN:label" - label is at the END!
    first = raw_lines[0]
    parts = first.split(':')
    dim_strs = [p.strip() for p in parts[:-1]]  # All except last (label)
    num_features = len(dim_strs)
    
    # Scan ALL samples to find true max_seq_len (for variable-length datasets)
    max_seq_len = 0
    for line in raw_lines:
        parts = line.split(':')
        dims = [p.strip() for p in parts[:-1]]
        for dim_str in dims:
            seq_len = len(dim_str.split(','))
            max_seq_len = max(max_seq_len, seq_len)
    
    print(f"  Samples: {num_samples}, Features: {num_features}, Seq len: {max_seq_len}")
    
    # Allocate arrays
    data_array = np.zeros((num_samples, max_seq_len, num_features), dtype=np.float32)
    labels = []
    
    # Parse all samples
    for idx, line in enumerate(raw_lines):
        parts = line.split(':')
        label = parts[-1].strip()  # Label is at the END!
        labels.append(label)
        
        dim_strs = [p.strip() for p in parts[:-1]]  # All except last (label)
        for j, dim_str in enumerate(dim_strs):
            values = dim_str.split(',')
            for t, v in enumerate(values):
                try:
                    data_array[idx, t, j] = float(v.strip())
                except:
                    data_array[idx, t, j] = 0.0
    
    return data_array, np.array(labels), max_seq_len, num_features


def generate_dataset_embeddings(
    dataset_name: str,
    data_dir: str | None = None,
    save_dir: str | None = None,
    splits: tuple[str, ...] = ("train", "test"),
    model_name: str = "gpt2",
    device: str = "cuda",
    use_simple: bool = False,
    use_statistics: bool = True,
    max_tokens: int = 896,
) -> None:
    """
    Generate embeddings for all splits of a classification dataset.
    
    Following TimeCMA's approach:
    - For each sample (T, D), process each dimension separately
    - Each dimension: time series -> Prompt -> GPT2 -> embedding (E,)
    - Concatenate: (E, D)
    
    Args:
        dataset_name: Name of the dataset
        data_dir: Directory containing the dataset
        save_dir: Directory to save embeddings
        splits: Tuple of splits to process
        model_name: LLM model name
        device: Device to use
        use_simple: Use simplified embedding generation
        use_statistics: True for statistical prompt, False for full sequence
        max_tokens: Max tokens for full sequence mode (auto-sampling if exceeded)
    """
    if data_dir is None:
        data_dir = str(PROJECT_ROOT / "datasets" / dataset_name)
    
    if save_dir is None:
        save_dir = str(PROJECT_ROOT / "Embeddings")
    
    print("=" * 60)
    print(f"Generating Embeddings for {dataset_name}")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"Save directory: {save_dir}")
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    if use_simple:
        print(f"Mode: Simple")
    else:
        mode_str = "Statistics" if use_statistics else f"Full Sequence (max_tokens={max_tokens})"
        print(f"Mode: {mode_str}")
    print("=" * 60)
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Import embedding generator from storage
    from gen_cls_emb import GenClassificationEmb, GenClassificationEmbSimple
    
    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device_obj}")
    
    for split in splits:
        print(f"\n{'='*40}")
        print(f"Processing {split} split...")
        print(f"{'='*40}")
        
        try:
            _generate_split_embeddings(
                dataset_name=dataset_name,
                data_dir=data_dir,
                save_dir=save_dir,
                split=split,
                model_name=model_name,
                device=device_obj,
                use_simple=use_simple,
                use_statistics=use_statistics,
                max_tokens=max_tokens,
            )
        except Exception as e:
            print(f"  Error processing {split}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "=" * 60)
    print("Embedding generation completed!")
    print(f"Embeddings saved to: {save_dir}/{dataset_name}/")
    print("=" * 60)


def _generate_split_embeddings(
    dataset_name: str,
    data_dir: str,
    save_dir: str,
    split: str,
    model_name: str,
    device: torch.device,
    use_simple: bool,
    use_statistics: bool = True,
    max_tokens: int = 896,
) -> None:
    """
    Generate embeddings for a single split.
    
    Following TimeCMA's exact approach:
    - Use GenClassificationEmb class (mirrors TimeCMA's GenPromptEmb)
    - Process sample by sample (batch_size=1, same as TimeCMA's store_emb.py)
    - For each sample (T, D):
      - For each dimension j in D:
        - Extract time series: (T,)
        - Create prompt based on use_statistics setting
        - Tokenize -> GPT2 -> last_hidden_state -> last token embedding (E,)
      - Stack all D embeddings -> (E, D)
    - Save to h5 file
    """
    from gen_cls_emb import GenClassificationEmb, GenClassificationEmbSimple
    
    # Load data
    data_array, labels, max_seq_len, num_features = load_ts_classification_data(
        data_dir, split
    )
    num_samples = data_array.shape[0]
    
    # Create save directory (remove stale files to ensure overwrite)
    save_path = os.path.join(save_dir, dataset_name, split)
    if os.path.exists(save_path):
        # Remove old embedding files
        old_files = glob.glob(os.path.join(save_path, "*.h5"))
        if old_files:
            print(f"  [overwrite] Removing {len(old_files)} old files...")
            for f in old_files:
                os.remove(f)
    os.makedirs(save_path, exist_ok=True)
    
    # Initialize embedding generator
    if use_simple:
        gen_emb = GenClassificationEmbSimple(
            dataset_name=dataset_name,
            model_name=model_name,
            device=device,
        ).to(device)
    else:
        gen_emb = GenClassificationEmb(
            dataset_name=dataset_name,
            model_name=model_name,
            device=device,
            seq_len=max_seq_len,
            use_statistics=use_statistics,
            max_tokens=max_tokens,
        ).to(device)
    
    gen_emb.eval()
    
    print(f"  Sample shape: ({max_seq_len}, {num_features})")
    print(f"  Output embedding shape per sample: (768, {num_features})")
    print(f"  Generating embeddings...")
    
    # Process sample by sample (batch_size=1, same as TimeCMA)
    with torch.no_grad():
        for idx in tqdm(range(num_samples), desc=f"  {split}"):
            # Get sample: (seq_len, num_features) -> (1, seq_len, num_features)
            sample_data = torch.tensor(
                data_array[idx], dtype=torch.float32
            ).unsqueeze(0).to(device)
            
            # Generate embeddings using GenClassificationEmb
            # Following TimeCMA's generate_embeddings:
            # - For each dimension j, create prompt and get GPT2 embedding
            # - Output: (1, E, N) where E=768, N=num_features
            embeddings = gen_emb.generate_embeddings(sample_data)
            
            # Remove batch dimension: (1, E, N) -> (E, N)
            embeddings = embeddings.squeeze(0).cpu().numpy()
            
            # Save to h5 file (same format as TimeCMA)
            file_path = os.path.join(save_path, f"{idx}.h5")
            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('embeddings', data=embeddings)
    
    print(f"  Saved {num_samples} embeddings to: {save_path}")


def generate_all_embeddings(
    datasets_dir: str | None = None,
    save_dir: str | None = None,
    use_simple: bool = False,
    use_statistics: bool = True,
    max_tokens: int = 896,
    device: str = "cuda",
) -> None:
    """Generate embeddings for all datasets in the datasets directory."""
    if datasets_dir is None:
        datasets_dir = str(PROJECT_ROOT / "datasets")
    
    if not os.path.exists(datasets_dir):
        raise FileNotFoundError(f"Datasets directory not found: {datasets_dir}")
    
    datasets = [d for d in os.listdir(datasets_dir) 
                if os.path.isdir(os.path.join(datasets_dir, d))]
    
    print(f"Found {len(datasets)} datasets: {datasets}")
    
    for i, dataset_name in enumerate(datasets, 1):
        print(f"\n[{i}/{len(datasets)}] Processing {dataset_name}...")
        try:
            generate_dataset_embeddings(
                dataset_name=dataset_name,
                save_dir=save_dir,
                use_simple=use_simple,
                use_statistics=use_statistics,
                max_tokens=max_tokens,
                device=device,
            )
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
            continue
