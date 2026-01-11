import argparse


class Options(object):

    def __init__(self):

        self.parser = argparse.ArgumentParser(
            description='TSCMA: TimeCMA for Time Series Classification')

        # Config file
        self.parser.add_argument('--config', dest='config_filepath',
                                 help='Configuration .json file (optional). Overwrites existing command-line args!')

        # TimeCMA model parameters
        self.parser.add_argument('--model_type', choices={'timecma', 'timecma_patch'}, 
                                 default='timecma_patch', help='TimeCMA model variant')
        self.parser.add_argument('--patch_size', type=int, default=8, help='Patch size for TimeCMA')
        self.parser.add_argument('--stride', type=int, default=8, help='Stride for patching')
        self.parser.add_argument('--d_model', type=int, default=768,
                                 help='Internal dimension of transformer embeddings')
        self.parser.add_argument('--d_llm', type=int, default=768,
                                 help='Dimension of LLM embeddings (768 for GPT-2)')
        self.parser.add_argument('--channel', type=int, default=64,
                                 help='Hidden channel dimension')
        self.parser.add_argument('--e_layers', type=int, default=2,
                                 help='Number of encoder layers')
        self.parser.add_argument('--n_heads', type=int, default=8,
                                 help='Number of attention heads')
        self.parser.add_argument('--d_ff', type=int, default=256,
                                 help='Feed-forward dimension')
        self.parser.add_argument('--dropout', type=float, default=0.1,
                                 help='Dropout rate')

        # Embedding parameters
        self.parser.add_argument('--use_embedding', action='store_true',
                                 help='Use pre-computed text embeddings from LLM (required for TimeCMA)')
        self.parser.add_argument('--embedding_dir', type=str, default='./Embeddings',
                                 help='Directory containing pre-computed embeddings')

        # I/O
        self.parser.add_argument('--output_dir', default='./output',
                                 help='Root output directory.')
        self.parser.add_argument('--data_dir', default='./data',
                                 help='Data directory')
        self.parser.add_argument('--load_model',
                                 help='Path to pre-trained model.')
        self.parser.add_argument('--resume', action='store_true',
                                 help='Resume training from checkpoint.')
        self.parser.add_argument('--change_output', action='store_true',
                                 help='Whether the loaded model will be fine-tuned on a different task')
        self.parser.add_argument('--save_all', action='store_true',
                                 help='Save model weights for every epoch')
        self.parser.add_argument('--name', dest='experiment_name', default='',
                                 help='Experiment name identifier')
        self.parser.add_argument('--comment', type=str, default='', help='Experiment comment/description')
        self.parser.add_argument('--no_timestamp', action='store_true',
                                 help='Do not append timestamp to output directory')
        self.parser.add_argument('--records_file', default='./records.xls',
                                 help='Excel file keeping all records of experiments')

        # System
        self.parser.add_argument('--console', action='store_true',
                                 help="Optimize printout for console output")
        self.parser.add_argument('--print_interval', type=int, default=1,
                                 help='Print batch info every this many batches')
        self.parser.add_argument('--gpu', type=str, default='0',
                                 help='GPU index, -1 for CPU')
        self.parser.add_argument('--n_proc', type=int, default=-1,
                                 help='Number of processes for data loading')
        self.parser.add_argument('--num_workers', type=int, default=0,
                                 help='Dataloader threads. 0 for single-thread.')
        self.parser.add_argument('--seed',
                                 help='Random seed for reproducibility')

        # Dataset
        self.parser.add_argument('--limit_size', type=float, default=None,
                                 help="Limit dataset size for debugging")
        self.parser.add_argument('--test_only', choices={'testset'},
                                 help='Only evaluate on test set')
        self.parser.add_argument('--data_class', type=str, default='tsra',
                                 help="Data type to process")
        self.parser.add_argument('--labels', type=str,
                                 help="Label column name(s) for multi-task")
        self.parser.add_argument('--test_from',
                                 help='Read test IDs from specified text file')
        self.parser.add_argument('--test_ratio', type=float, default=0,
                                 help="Test set proportion")
        self.parser.add_argument('--val_ratio', type=float, default=0.2,
                                 help="Validation set proportion")
        self.parser.add_argument('--pattern', type=str,
                                 help='Regex pattern for training data files')
        self.parser.add_argument('--val_pattern', type=str,
                                 help="Regex pattern for validation data files")
        self.parser.add_argument('--test_pattern', type=str,
                                 help="Regex pattern for test data files")
        self.parser.add_argument('--normalization',
                                 choices={'standardization', 'minmax', 'per_sample_std', 'per_sample_minmax'},
                                 default='standardization',
                                 help='Normalization method')
        self.parser.add_argument('--norm_from',
                                 help="Load normalization values from pickle file")
        self.parser.add_argument('--subsample_factor', type=int, default=None,
                                 help="Factor for subsampling time series (optional)")

        # Training
        self.parser.add_argument('--task', choices={"classification", "regression"},
                                 default="classification",
                                 help="Training task: classification or regression")
        self.parser.add_argument('--epochs', type=int, default=50,
                                 help='Number of training epochs')
        self.parser.add_argument('--val_interval', type=int, default=2,
                                 help='Evaluate on validation set every this many epochs')
        self.parser.add_argument('--optimizer', choices={"Adam", "RAdam"}, default="RAdam", 
                                 help="Optimizer")
        self.parser.add_argument('--lr', type=float, default=5e-4,
                                 help='Learning rate')
        self.parser.add_argument('--lr_step', type=str, default='1000000',
                                 help='Epochs when to reduce learning rate')
        self.parser.add_argument('--lr_factor', type=str, default='0.1',
                                 help="Learning rate decay factors")
        self.parser.add_argument('--batch_size', type=int, default=64,
                                 help='Training batch size')
        self.parser.add_argument('--test_batch_size', type=int, default=1,
                                 help='Evaluation/test batch size (default=1 to avoid drop_last issues)')
        self.parser.add_argument('--l2_reg', type=float, default=0,
                                 help='L2 weight regularization')
        self.parser.add_argument('--global_reg', action='store_true',
                                 help='Apply L2 regularization to all weights')
        self.parser.add_argument('--key_metric', choices={'loss', 'accuracy', 'precision'}, default='accuracy',
                                 help='Metric for best epoch selection')
        self.parser.add_argument('--freeze', action='store_true',
                                 help='Freeze all layers except output layer')

        # Model architecture (kept for compatibility)
        self.parser.add_argument('--max_seq_len', type=int,
                                 help="Maximum input sequence length")
        self.parser.add_argument('--activation', choices={'relu', 'gelu'}, default='gelu',
                                 help='Activation function')

    def parse(self):

        args = self.parser.parse_args()

        args.lr_step = [int(i) for i in args.lr_step.split(',')]
        args.lr_factor = [float(i) for i in args.lr_factor.split(',')]
        if (len(args.lr_step) > 1) and (len(args.lr_factor) == 1):
            args.lr_factor = len(args.lr_step) * args.lr_factor

        assert len(args.lr_step) == len(args.lr_factor), \
            "You must specify as many values in `lr_step` as in `lr_factors`"

        if args.val_pattern is not None:
            args.val_ratio = 0
            args.test_ratio = 0

        return args
