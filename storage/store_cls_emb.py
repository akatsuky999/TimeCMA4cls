"""
Store and manage embeddings for classification datasets.

This script generates and stores GPT-2 embeddings for time series
classification data, similar to TimeCMA's approach for forecasting.
"""

import os
import sys
import argparse
import time
import h5py
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from gen_cls_emb import GenClassificationEmb, GenClassificationEmbSimple


class EmbeddingGenerator:
    """
    Embedding generator for classification datasets.
    
    Generates and stores embeddings in h5 format for efficient loading
    during training.
    """
    
    def __init__(
        self,
        dataset_name: str,
        model_name: str = "gpt2",
        device: str = "cuda",
        d_model: int = 768,
        use_simple: bool = False,
        use_statistics: bool = True,
        max_tokens: int = 896,
    ):
        """
        Initialize embedding generator.
        
        Args:
            dataset_name: Name of the dataset
            model_name: Name of the LLM model (default: gpt2)
            device: Device to use for computation
            d_model: Model embedding dimension
            use_simple: Use simplified (faster) embedding generation
            use_statistics: True for statistical prompt, False for full sequence
            max_tokens: Max tokens for full sequence mode (auto-sampling if exceeded)
        """
        self.dataset_name = dataset_name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.d_model = d_model
        
        # Initialize embedding generator
        if use_simple:
            self.gen_emb = GenClassificationEmbSimple(
                model_name=model_name,
                device=self.device,
                d_model=d_model
            )
        else:
            self.gen_emb = GenClassificationEmb(
                dataset_name=dataset_name,
                model_name=model_name,
                device=self.device,
                d_model=d_model,
                use_statistics=use_statistics,
                max_tokens=max_tokens,
            )
        
        self.gen_emb.to(self.device)
        self.gen_emb.eval()
    
    def generate_and_save(
        self,
        data_array: np.ndarray,
        save_dir: str,
        split_name: str = "train",
        batch_size: int = 1
    ):
        """
        Generate embeddings and save to files.
        
        Args:
            data_array: Data array of shape [num_samples, seq_len, num_features]
            save_dir: Directory to save embeddings
            split_name: Name of the split (train/test/val)
            batch_size: Batch size for processing
        """
        # Create save directory
        save_path = os.path.join(save_dir, self.dataset_name, split_name)
        os.makedirs(save_path, exist_ok=True)
        
        num_samples = data_array.shape[0]
        
        print(f"Generating embeddings for {num_samples} samples...")
        
        with torch.no_grad():
            for i in tqdm(range(0, num_samples, batch_size), desc=f"Processing {split_name}"):
                # Get batch
                batch_end = min(i + batch_size, num_samples)
                batch_data = torch.tensor(
                    data_array[i:batch_end], 
                    dtype=torch.float32
                ).to(self.device)
                
                # Generate embeddings: [B, E, N]
                embeddings = self.gen_emb.generate_embeddings(batch_data)
                
                # Save each sample individually
                for j, idx in enumerate(range(i, batch_end)):
                    emb = embeddings[j].cpu().numpy()  # [E, N]
                    
                    # Save to h5 file
                    file_path = os.path.join(save_path, f"{idx}.h5")
                    with h5py.File(file_path, 'w') as hf:
                        hf.create_dataset('embeddings', data=emb)
        
        print(f"Saved embeddings to {save_path}")
        return save_path


def save_embeddings(
    feature_df,
    all_ids,
    dataset_name: str,
    max_seq_len: int,
    save_dir: str = "./Embeddings",
    split_name: str = "train",
    model_name: str = "gpt2",
    device: str = "cuda",
    use_simple: bool = True,
    use_statistics: bool = True,
    max_tokens: int = 896,
    batch_size: int = 1
):
    """
    Generate and save embeddings for a dataset.
    
    This is the main function to call for embedding generation.
    
    Args:
        feature_df: Feature DataFrame from the dataset
        all_ids: List of sample IDs
        dataset_name: Name of the dataset
        max_seq_len: Maximum sequence length
        save_dir: Directory to save embeddings
        split_name: Name of the split (train/test/val)
        model_name: Name of the LLM model
        device: Device to use
        use_simple: Use simplified embedding generation
        use_statistics: True for statistical prompt, False for full sequence
        max_tokens: Max tokens for full sequence mode
        batch_size: Batch size for processing
    """
    # Convert DataFrame to numpy array
    # feature_df is indexed by sample ID with multiple rows per sample
    num_samples = len(all_ids)
    num_features = feature_df.shape[1]
    
    print(f"Preparing data for {num_samples} samples with {num_features} features...")
    
    # Reshape data: [num_samples, seq_len, num_features]
    data_list = []
    for sample_id in tqdm(all_ids, desc="Loading samples"):
        sample_data = feature_df.loc[sample_id].values
        if len(sample_data.shape) == 1:
            # Single row sample
            sample_data = sample_data.reshape(1, -1)
        # Pad or truncate to max_seq_len
        if sample_data.shape[0] < max_seq_len:
            padding = np.zeros((max_seq_len - sample_data.shape[0], num_features))
            sample_data = np.vstack([sample_data, padding])
        elif sample_data.shape[0] > max_seq_len:
            sample_data = sample_data[:max_seq_len]
        data_list.append(sample_data)
    
    data_array = np.stack(data_list, axis=0)  # [num_samples, seq_len, num_features]
    
    # Initialize generator and generate embeddings
    generator = EmbeddingGenerator(
        dataset_name=dataset_name,
        model_name=model_name,
        device=device,
        use_simple=use_simple,
        use_statistics=use_statistics,
        max_tokens=max_tokens,
    )
    
    save_path = generator.generate_and_save(
        data_array=data_array,
        save_dir=save_dir,
        split_name=split_name,
        batch_size=batch_size
    )
    
    return save_path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate embeddings for classification dataset")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the dataset")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset")
    parser.add_argument("--save_dir", type=str, default="./Embeddings", help="Directory to save embeddings")
    parser.add_argument("--model_name", type=str, default="gpt2", help="LLM model name")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--use_simple", action="store_true", help="Use simplified embedding generation")
    parser.add_argument("--use_statistics", action="store_true", default=True, 
                        help="Use statistical prompt (True) or full sequence (False)")
    parser.add_argument("--use_full_sequence", action="store_true", 
                        help="Use full sequence prompt instead of statistics")
    parser.add_argument("--max_tokens", type=int, default=896, help="Max tokens for full sequence mode")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test", "val"], help="Data split")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    print(f"Generating embeddings for {args.dataset_name}...")
    print(f"Data directory: {args.data_dir}")
    print(f"Save directory: {args.save_dir}")
    print(f"Using model: {args.model_name}")
    print(f"Device: {args.device}")
    print(f"Simple mode: {args.use_simple}")
    
    # This is a standalone script - actual data loading should be done
    # through the main training script which has access to the data classes
    print("\nNote: This script should be called from the main training pipeline")
    print("which has access to the loaded dataset. See main.py for integration.")

