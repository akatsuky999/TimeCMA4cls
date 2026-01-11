"""
Generate embeddings for all datasets.

Usage:
    cd ClassCMA
    python config_emb/gen_all.py
"""

from emb_utils import generate_all_embeddings

if __name__ == "__main__":
    generate_all_embeddings(
        use_simple=False,
        use_statistics=True,  # True: summary statistics, False: full sequence
        max_tokens=896,       # Max token count when running full-sequence mode
        device="cuda",
    )
