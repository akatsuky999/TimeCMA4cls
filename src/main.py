"""
TSCMA: TimeCMA for Time Series Classification

Main training script for TimeCMA model.
"""

import logging

logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Loading packages ...")
import os
import sys
import time
import pickle
import json

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from options import Options
from running import setup, pipeline_factory, validate, check_progress, NEG_METRICS
from utils import utils
from datasets.data import data_factory, Normalizer
from datasets.datasplit import split_dataset
from models.timecma_cls import TimeCMAClassifier, TimeCMAClassifierWithPatching
from models.loss import get_loss_module
from optimizers import get_optimizer


def main(config):

    total_epoch_time = 0
    total_eval_time = 0

    total_start_time = time.time()

    # Add file logging besides stdout
    file_handler = logging.FileHandler(os.path.join(config['output_dir'], 'output.log'))
    logger.addHandler(file_handler)

    logger.info('Running:\n{}\n'.format(' '.join(sys.argv)))

    if config['seed'] is not None:
        torch.manual_seed(config['seed'])

    device = torch.device('cuda' if (torch.cuda.is_available() and config['gpu'] != '-1') else 'cpu')
    logger.info("Using device: {}".format(device))
    if device == 'cuda':
        logger.info("Device index: {}".format(torch.cuda.current_device()))

    # Build data
    logger.info("Loading and preprocessing data ...")
    data_class = data_factory[config['data_class']]
    my_data = data_class(config['data_dir'], pattern=config['pattern'], n_proc=config['n_proc'], limit_size=config['limit_size'], config=config)
    feat_dim = my_data.feature_df.shape[1]
    
    if config['task'] == 'classification':
        validation_method = 'StratifiedShuffleSplit'
        labels = my_data.labels_df.values.flatten()
    else:
        validation_method = 'ShuffleSplit'
        labels = None

    # Split dataset
    test_data = my_data
    test_indices = None
    val_data = my_data
    val_indices = []
    
    if config['test_pattern']:
        test_data = data_class(config['data_dir'], pattern=config['test_pattern'], n_proc=-1, config=config)
        test_indices = test_data.all_IDs
    if config['test_from']:
        test_indices = list(set([line.rstrip() for line in open(config['test_from']).readlines()]))
        try:
            test_indices = [int(ind) for ind in test_indices]
        except ValueError:
            pass
        logger.info("Loaded {} test IDs from file: '{}'".format(len(test_indices), config['test_from']))
    if config['val_pattern']:
        val_data = data_class(config['data_dir'], pattern=config['val_pattern'], n_proc=-1, config=config)
        val_indices = val_data.all_IDs

    if config['val_ratio'] > 0:
        train_indices, val_indices, test_indices = split_dataset(data_indices=my_data.all_IDs,
                                                                 validation_method=validation_method,
                                                                 n_splits=1,
                                                                 validation_ratio=config['val_ratio'],
                                                                 test_set_ratio=config['test_ratio'],
                                                                 test_indices=test_indices,
                                                                 random_seed=1337,
                                                                 labels=labels)
        train_indices = train_indices[0]
        val_indices = val_indices[0]
    else:
        train_indices = my_data.all_IDs
        if test_indices is None:
            test_indices = []

    logger.info("{} samples may be used for training".format(len(train_indices)))
    logger.info("{} samples will be used for validation".format(len(val_indices)))
    logger.info("{} samples will be used for testing".format(len(test_indices)))

    with open(os.path.join(config['output_dir'], 'data_indices.json'), 'w') as f:
        try:
            json.dump({'train_indices': list(map(int, train_indices)),
                       'val_indices': list(map(int, val_indices)),
                       'test_indices': list(map(int, test_indices))}, f, indent=4)
        except ValueError:
            json.dump({'train_indices': list(train_indices),
                       'val_indices': list(val_indices),
                       'test_indices': list(test_indices)}, f, indent=4)

    # Pre-process features
    normalizer = None
    if config['norm_from']:
        with open(config['norm_from'], 'rb') as f:
            norm_dict = pickle.load(f)
        normalizer = Normalizer(**norm_dict)
    elif config['normalization'] is not None:
        normalizer = Normalizer(config['normalization'])
        my_data.feature_df.loc[train_indices] = normalizer.normalize(my_data.feature_df.loc[train_indices])
        if not config['normalization'].startswith('per_sample'):
            norm_dict = normalizer.__dict__
            with open(os.path.join(config['output_dir'], 'normalization.pickle'), 'wb') as f:
                pickle.dump(norm_dict, f, pickle.HIGHEST_PROTOCOL)
    if normalizer is not None:
        if len(val_indices):
            val_data.feature_df.loc[val_indices] = normalizer.normalize(val_data.feature_df.loc[val_indices])
        if len(test_indices):
            test_data.feature_df.loc[test_indices] = normalizer.normalize(test_data.feature_df.loc[test_indices])

    # Create TimeCMA model
    logger.info("Creating TimeCMA model ...")
    model_type = config.get('model_type', 'timecma_patch')
    
    if model_type == 'timecma':
        logger.info("Using TimeCMA model (without patching)")
        model = TimeCMAClassifier(config, my_data)
    else:  # timecma_patch (default)
        logger.info("Using TimeCMA model with patching")
        model = TimeCMAClassifierWithPatching(config, my_data)

    if config['freeze']:
        for name, param in model.named_parameters():
            if name.startswith('classifier'):
                param.requires_grad = True
            else:
                param.requires_grad = False

    logger.info("Model:\n{}".format(model))
    logger.info("Total number of parameters: {}".format(utils.count_parameters(model)))
    logger.info("Trainable parameters: {}".format(utils.count_parameters(model, trainable=True)))

    # Initialize optimizer
    if config['global_reg']:
        weight_decay = config['l2_reg']
        output_reg = None
    else:
        weight_decay = 0
        output_reg = config['l2_reg']

    optim_class = get_optimizer(config['optimizer'])
    optimizer = optim_class(model.parameters(), lr=config['lr'], weight_decay=weight_decay)

    start_epoch = 0
    lr_step = 0
    lr = config['lr']
    
    # Load model and optimizer state
    if args.load_model:
        model, optimizer, start_epoch = utils.load_model(model, config['load_model'], optimizer, config['resume'],
                                                         config['change_output'],
                                                         config['lr'],
                                                         config['lr_step'],
                                                         config['lr_factor'])
    model.to(device)

    loss_module = get_loss_module(config)

    # Evaluation batch size (default=1 to avoid drop_last issues)
    eval_batch_size = config.get('test_batch_size', 1)
    
    if config['test_only'] == 'testset':
        dataset_class, collate_fn, runner_class = pipeline_factory(config)
        test_dataset = dataset_class(test_data, test_indices, split_name='test')

        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=eval_batch_size,
                                 shuffle=False,
                                 num_workers=config['num_workers'],
                                 pin_memory=True,
                                 collate_fn=lambda x: collate_fn(x, max_len=model.max_len))
        test_evaluator = runner_class(model, test_loader, device, loss_module,
                                            print_interval=config['print_interval'], console=config['console'])
        aggr_metrics_test, per_batch_test = test_evaluator.evaluate(keep_all=True)
        print_str = 'Test Summary: '
        for k, v in aggr_metrics_test.items():
            print_str += '{}: {:8f} | '.format(k, v)
        logger.info(print_str)
        return
    
    # Initialize data generators
    dataset_class, collate_fn, runner_class = pipeline_factory(config)
    
    # TimeCMA always uses embeddings
    # Note: validation uses TEST pattern data, so split_name='test' for embeddings
    val_dataset = dataset_class(val_data, val_indices, split_name='test')
    train_dataset = dataset_class(my_data, train_indices, split_name='train')

    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=eval_batch_size,  # Use batch_size=1 to avoid drop_last issues
                            shuffle=False,
                            num_workers=config['num_workers'],
                            pin_memory=True,
                            collate_fn=lambda x: collate_fn(x, max_len=model.max_len))

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=config['batch_size'],
                              shuffle=True,
                              num_workers=config['num_workers'],
                              pin_memory=True,
                              collate_fn=lambda x: collate_fn(x, max_len=model.max_len))

    trainer = runner_class(model, train_loader, device, loss_module, optimizer, l2_reg=output_reg,
                                 print_interval=config['print_interval'], console=config['console'])
    val_evaluator = runner_class(model, val_loader, device, loss_module,
                                       print_interval=config['print_interval'], console=config['console'])

    tensorboard_writer = SummaryWriter(config['tensorboard_dir'])

    best_value = 1e16 if config['key_metric'] in NEG_METRICS else -1e16
    metrics = []
    best_metrics = {}
    best_epoch = 0
    
    # Training history for simple logging
    training_history = []

    # Evaluate on validation before training
    aggr_metrics_val, best_metrics, best_value = validate(val_evaluator, tensorboard_writer, config, best_metrics,
                                                          best_value, epoch=0)
    metrics_names, metrics_values = zip(*aggr_metrics_val.items())
    metrics.append(list(metrics_values))

    logger.info('Starting training...')
    for epoch in range(start_epoch + 1, config["epochs"] + 1):
        mark = epoch if config['save_all'] else 'last'
        epoch_start_time = time.time()
        aggr_metrics_train = trainer.train_epoch(epoch)
        epoch_runtime = time.time() - epoch_start_time
        total_epoch_time += epoch_runtime
        
        # Log training metrics to tensorboard
        for k, v in aggr_metrics_train.items():
            tensorboard_writer.add_scalar('{}/train'.format(k), v, epoch)
        
        # Concise epoch summary (single line)
        logger.info(f"Epoch {epoch}/{config['epochs']} | Train Loss: {aggr_metrics_train.get('loss', 0):.4f} | Time: {epoch_runtime:.1f}s")

        # evaluate if first or last epoch or at specified interval
        if (epoch == config["epochs"]) or (epoch == start_epoch + 1) or (epoch % config['val_interval'] == 0):
            prev_best = best_value
            aggr_metrics_val, best_metrics, best_value = validate(val_evaluator, tensorboard_writer, config,
                                                                  best_metrics, best_value, epoch)
            metrics_names, metrics_values = zip(*aggr_metrics_val.items())
            metrics.append(list(metrics_values))
            
            # Track best epoch
            if best_value != prev_best:
                best_epoch = epoch
            
            # Record training history (concise format)
            history_entry = {
                'epoch': epoch,
                'train_loss': aggr_metrics_train.get('loss', 0),
                'val_loss': aggr_metrics_val.get('loss', 0),
                'val_accuracy': aggr_metrics_val.get('accuracy', 0),
                'val_precision': aggr_metrics_val.get('precision', 0),
            }
            training_history.append(history_entry)

        utils.save_model(os.path.join(config['save_dir'], 'model_{}.pth'.format(mark)), epoch, model, optimizer)

        # Learning rate scheduling
        if epoch == config['lr_step'][lr_step]:
            utils.save_model(os.path.join(config['save_dir'], 'model_{}.pth'.format(epoch)), epoch, model, optimizer)
            lr = lr * config['lr_factor'][lr_step]
            if lr_step < len(config['lr_step']) - 1:
                lr_step += 1
            logger.info('Learning rate updated to: ', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        # Difficulty scheduling
        if config.get('harden') and check_progress(epoch):
            train_loader.dataset.update()
            val_loader.dataset.update()

    # Export evolution of metrics over epochs
    header = metrics_names
    metrics_filepath = os.path.join(config["output_dir"], "metrics_" + config["experiment_name"] + ".xls")
    book = utils.export_performance_metrics(metrics_filepath, metrics, header, sheet_name="metrics")

    # Export record metrics to a file accumulating records from all experiments
    utils.register_record(config["records_file"], config["initial_timestamp"], config["experiment_name"],
                          best_metrics, aggr_metrics_val, comment=config['comment'])

    total_runtime = time.time() - total_start_time
    
    # Save concise training summary
    training_summary = {
        'experiment': config['experiment_name'],
        'dataset': os.path.basename(config['data_dir']),
        'model_type': config.get('model_type', 'timecma_patch'),
        'total_epochs': config['epochs'],
        'best_epoch': best_epoch,
        'best_accuracy': best_metrics.get('accuracy', 0),
        'best_precision': best_metrics.get('precision', 0),
        'best_loss': best_metrics.get('loss', 0),
        'final_accuracy': aggr_metrics_val.get('accuracy', 0),
        'runtime_seconds': round(total_runtime, 2),
        'history': training_history
    }
    
    summary_path = os.path.join(config['output_dir'], 'training_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(training_summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Training summary saved to: {summary_path}")
    
    # Print concise summary
    logger.info('\n' + '='*50)
    logger.info('TRAINING SUMMARY')
    logger.info('='*50)
    logger.info(f"Dataset: {training_summary['dataset']}")
    logger.info(f"Best Epoch: {best_epoch}")
    logger.info(f"Best Accuracy: {best_metrics.get('accuracy', 0):.4f}")
    logger.info(f"Best Precision: {best_metrics.get('precision', 0):.4f}")
    logger.info(f"Final Accuracy: {aggr_metrics_val.get('accuracy', 0):.4f}")
    logger.info(f"Total Runtime: {utils.readable_time(total_runtime)[0]}h {utils.readable_time(total_runtime)[1]}m {utils.readable_time(total_runtime)[2]:.1f}s")
    logger.info('='*50)
    
    logger.info('All Done!')

    return best_value


if __name__ == '__main__':

    args = Options().parse()
    config = setup(args)
    main(config)
