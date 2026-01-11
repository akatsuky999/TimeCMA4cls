"""
Generate embeddings for JapaneseVowels dataset.

Usage:
    cd ClassCMA
    python config_emb/gen_JapaneseVowels.py
"""

from emb_utils import generate_dataset_embeddings

if __name__ == "__main__":
    generate_dataset_embeddings(
        dataset_name="JapaneseVowels",
        splits=("train", "test"),
        use_simple=False,
        use_statistics=False,  # True: summary statistics, False: full sequence
        max_tokens=896,       # Max token count when running full-sequence mode
        device="cuda",
    )
