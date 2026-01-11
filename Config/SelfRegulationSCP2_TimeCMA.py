"""
Run TimeCMA classification experiments on the SelfRegulationSCP2 dataset.

Steps:
1. Generate GPT-2 embeddings: python config_emb/gen_SelfRegulationSCP2.py
2. Run training: python Config/SelfRegulationSCP2_TimeCMA.py
"""

from run_utils import run_timecma_experiments

if __name__ == "__main__":
    print("=" * 60)
    print("TimeCMA Classification - SelfRegulationSCP2 Dataset")
    print("Using GPT-2 embeddings from ./Embeddings")
    print("=" * 60)
    
    run_timecma_experiments(
        experiment_name="SelfRegulationSCP2-TimeCMA",
        data_subdir="SelfRegulationSCP2",
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
