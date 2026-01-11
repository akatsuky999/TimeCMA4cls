"""
Generate embeddings for SelfRegulationSCP1 dataset.

Usage:
    cd ClassCMA
    python config_emb/gen_SelfRegulationSCP1.py
"""

from emb_utils import generate_dataset_embeddings

if __name__ == "__main__":
    generate_dataset_embeddings(
        dataset_name="SelfRegulationSCP1",
        splits=("train", "test"),
        use_simple=False,
        use_statistics=False,  # True: summary statistics, False: full sequence
        max_tokens=896,       # Max token count when running full-sequence mode
        device="cuda",
    )
