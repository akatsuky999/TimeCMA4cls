"""
Run TimeCMA classification experiments on the Heartbeat dataset.

Steps:
1. Generate GPT-2 embeddings: python config_emb/gen_Heartbeat.py
2. Run training: python Config/Heartbeat_TimeCMA.py
"""

from run_utils import run_timecma_experiments

if __name__ == "__main__":
    print("=" * 60)
    print("TimeCMA Classification - Heartbeat Dataset")
    print("Using GPT-2 embeddings from ./Embeddings")
    print("=" * 60)
    
    run_timecma_experiments(
        experiment_name="Heartbeat-TimeCMA",
        data_subdir="Heartbeat",
        lr_values=[0.0005],
        channel_values=[64],
        e_layers_values=[2],
        n_heads_values=[8],
        d_ff_values=[64],
        model_type='timecma_patch',
        use_embedding=True,
        epochs=80,
        seed=668,
    )
