"""
Run TimeCMA classification on the EthanolConcentration dataset.

Steps:
1. Generate GPT-2 embeddings: python config_emb/gen_EthanolConcentration.py
2. Start training: python Config/EthanolConcentration_TimeCMA.py
"""

from run_utils import run_timecma_experiments

if __name__ == "__main__":
    print("=" * 60)
    print("TimeCMA Classification - EthanolConcentration Dataset")
    print("Using GPT-2 embeddings from ./Embeddings")
    print("=" * 60)
    
    run_timecma_experiments(
        experiment_name="EthanolConcentration-TimeCMA",
        data_subdir="EthanolConcentration",
        lr_values=[0.0005],
        channel_values=[64],
        e_layers_values=[2],
        n_heads_values=[8],
        d_ff_values=[64],
        model_type='timecma_patch',
        use_embedding=True,
        epochs=40,
        seed=667,
    )
