"""
Generate embeddings for Heartbeat dataset.

Usage:
    cd ClassCMA
    python config_emb/gen_Heartbeat.py
"""

from emb_utils import generate_dataset_embeddings

if __name__ == "__main__":
    generate_dataset_embeddings(
        dataset_name="Heartbeat",
        splits=("train", "test"),
        use_simple=False,
        use_statistics=False,  # True: summary statistics, False: full sequence
        max_tokens=896,       # Max token count when running full-sequence mode
        device="cuda",
    )
