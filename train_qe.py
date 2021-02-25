#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
"""
Train a new model on one or across multiple GPUs.
"""

import collections
import math
import os
import random
import numpy as np

from fairseq.xml.utils import AttrDict
from fairseq.xml.data.dictionary import Dictionary, BOS_WORD, EOS_WORD, PAD_WORD, UNK_WORD, MASK_WORD
from fairseq.xml.model.transformer import TransformerModel
import torch.nn.functional as F

import torch

from fairseq import checkpoint_utils, distributed_utils, options, progress_bar, tasks, utils
from fairseq.data import iterators
from fairseq.mtqe.trainer_qe import Trainer
from fairseq.meters import AverageMeter, StopwatchMeter


def main(args, init_distributed=False):
    utils.import_user_module(args)

    ########### xml initialization
    ##  xml_model/mlm_tlm_xnli15_1024.pth
    reloaded = torch.load(args.xml_model_path)
    params = AttrDict(reloaded['params'])
    print("Supported languages: %s" % ", ".join(params.lang2id.keys()))

    # build dictionary / update parameters
    xml_dico = Dictionary(reloaded['dico_id2word'], reloaded['dico_word2id'], reloaded['dico_counts'])
    params.n_words = len(xml_dico)
    params.bos_index = xml_dico.index(BOS_WORD)
    params.eos_index = xml_dico.index(EOS_WORD)
    params.pad_index = xml_dico.index(PAD_WORD)
    params.unk_index = xml_dico.index(UNK_WORD)
    params.mask_index = xml_dico.index(MASK_WORD)

    xml_model = TransformerModel(params, xml_dico, True, True)
    xml_model.eval()
    print(xml_model._modules['position_embeddings']._parameters['weight'][0, 0].item())
    xml_model.load_state_dict(reloaded['model'])
    print(xml_model._modules['position_embeddings']._parameters['weight'][0, 0].item())
    ############## end of xml initilization

    assert args.max_tokens is not None or args.max_sentences is not None, \
        'Must specify batch size either with --max-tokens or --max-sentences'

    # Initialize CUDA and distributed training
    if torch.cuda.is_available() and not args.cpu:
        torch.cuda.set_device(args.device_id)
    torch.manual_seed(args.seed)
    if init_distributed:
        args.distributed_rank = distributed_utils.distributed_init(args)

    if distributed_utils.is_master(args):
        checkpoint_utils.verify_checkpoint_directory(args.save_dir)

    # Print args
    print(args)

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(args)

    # Load valid dataset (we load training data below, based on the latest checkpoint)
    for valid_sub_split in args.valid_subset.split(','):
        task.load_dataset(valid_sub_split, combine=False, shuffle=False, epoch=0, xml_dico=xml_dico, xml_params=params, task=args.task)

    # Build model and criterion
    predictor, estimator, xml_estmimator = task.build_model(args)
    # xml_estmimator = None

    for n, p in predictor.named_parameters():
        if not n.startswith('estimator'):
            p.requires_grad = False

    criterion = task.build_criterion(args)
    print(predictor)
    print('| model {}, criterion {}'.format(args.arch, criterion.__class__.__name__))
    print('| num. model params: {} (num. trained: {})'.format(
        sum(p.numel() for p in predictor.parameters()),
        sum(p.numel() for p in predictor.parameters() if p.requires_grad),
    ))

    # Build trainer
    trainer = Trainer(args, task, predictor, estimator, criterion, xml_model=xml_model, xml_estmimator=xml_estmimator)
    print('| training on {} GPUs'.format(args.distributed_world_size))
    print('| max tokens per GPU = {} and max sentences per GPU = {}'.format(
        args.max_tokens,
        args.max_sentences,
    ))

    # Load the latest checkpoint if one is available and restore the
    # corresponding train iterator
    print(predictor._modules['encoder']._modules['embed_tokens']._parameters['weight'].data[0, 0].item())
    checkpoint_utils.load_predictor_checkpoint(args, trainer)
    print(predictor._modules['encoder']._modules['embed_tokens']._parameters['weight'].data[0, 0].item())

    extra_state, epoch_itr = checkpoint_utils.load_estimator_checkpoint(args, trainer, xml_dico=xml_dico, xml_params=params)

    # Train until the learning rate gets too small
    max_epoch = args.max_epoch or math.inf
    max_update = args.max_update or math.inf
    lr = trainer.get_lr()
    train_meter = StopwatchMeter()
    train_meter.start()
    valid_losses = [None]
    valid_subsets = args.valid_subset.split(',')

    if args.evaluate > 0:
        validate(args, trainer, task, epoch_itr, valid_subsets, evaluate=True)
        return

    while lr > args.min_lr and epoch_itr.epoch < max_epoch and trainer.get_num_updates() < max_update:
        # train for one epoch
        train_qe(args, trainer, task, epoch_itr)

        # if not args.disable_validation and epoch_itr.epoch % args.validate_interval == 0:
        if not args.disable_validation:
            valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets)
        else:
            valid_losses = [None]

        # only use first validation loss to update the learning rate
        lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])

        # save checkpoint
        # if epoch_itr.epoch % args.save_interval == 0:
        #     checkpoint_utils.save_checkpoint(args, trainer, epoch_itr, valid_losses[0])

        if ':' in getattr(args, 'data', ''):
            # sharded data: get train iterator for next epoch
            epoch_itr = trainer.get_train_iterator(epoch_itr.epoch)
    train_meter.stop()
    print('| done training in {:.1f} seconds'.format(train_meter.sum))

def train_qe(args, trainer, task, epoch_itr):
    """Train the model for one epoch."""
    # Update parameters every N batches
    update_freq = args.update_freq[epoch_itr.epoch - 1] \
        if epoch_itr.epoch <= len(args.update_freq) else args.update_freq[-1]

    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=args.fix_batches_to_gpus,
        shuffle=(epoch_itr.epoch >= args.curriculum),
    )
    itr = iterators.GroupedIterator(itr, update_freq)
    progress = progress_bar.build_progress_bar(
        args, itr, epoch_itr.epoch, no_progress_bar='simple',
    )

    extra_meters = collections.defaultdict(lambda: AverageMeter())
    valid_subsets = args.valid_subset.split(',')



    max_update = args.max_update or math.inf
    for i, samples in enumerate(progress, start=epoch_itr.iterations_in_epoch):
        log_output = trainer.train_step(samples)
        if log_output is None:
            continue

        # log mid-epoch stats
        stats = get_training_stats(trainer)

        for k, v in log_output.items():
            if k in ['loss', 'nll_loss', 'ntokens', 'nsentences', 'sample_size', 'posterior']:
                continue  # these are already logged above
            if 'loss' in k or k == 'accuracy':
                extra_meters[k].update(v, log_output['sample_size'])
            else:
                extra_meters[k].update(v)
            stats[k] = extra_meters[k].avg
        progress.log(stats, tag='train', step=stats['num_updates'])

        # ignore the first mini-batch in words-per-second calculation
        if i == 0:
            trainer.get_meter('wps').reset()

        num_updates = trainer.get_num_updates()
        if (
            not args.disable_validation
            and args.save_interval_updates > 0
            # and num_updates % args.save_interval_updates == 0
            and num_updates % 50 == 0
            and num_updates > 0
        ):
            valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets)
            print(stats['lr'])
            checkpoint_utils.save_checkpoint(args, trainer, epoch_itr, valid_losses[0])

        if num_updates >= max_update:
            break

    # log end-of-epoch stats
    stats = get_training_stats(trainer)
    for k, meter in extra_meters.items():
        stats[k] = meter.avg
    progress.print(stats, tag='train', step=stats['num_updates'])

    # reset training meters
    for k in [
        'train_loss', 'train_nll_loss', 'wps', 'ups', 'wpb', 'bsz', 'gnorm', 'clip',
    ]:
        meter = trainer.get_meter(k)
        if meter is not None:
            meter.reset()

    return 1


def train(args, trainer, task, epoch_itr):
    """Train the model for one epoch."""
    # Update parameters every N batches
    update_freq = args.update_freq[epoch_itr.epoch - 1] \
        if epoch_itr.epoch <= len(args.update_freq) else args.update_freq[-1]

    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=args.fix_batches_to_gpus,
        shuffle=(epoch_itr.epoch >= args.curriculum),
    )
    itr = iterators.GroupedIterator(itr, update_freq)
    progress = progress_bar.build_progress_bar(
        args, itr, epoch_itr.epoch, no_progress_bar='simple',
    )

    extra_meters = collections.defaultdict(lambda: AverageMeter())
    valid_subsets = args.valid_subset.split(',')
    max_update = args.max_update or math.inf
    for i, samples in enumerate(progress, start=epoch_itr.iterations_in_epoch):
        log_output = trainer.train_step(samples)
        if log_output is None:
            continue

        # log mid-epoch stats
        stats = get_training_stats(trainer)
        for k, v in log_output.items():
            if k in ['loss', 'nll_loss', 'ntokens', 'nsentences', 'sample_size']:
                continue  # these are already logged above
            if 'loss' in k or k == 'accuracy':
                extra_meters[k].update(v, log_output['sample_size'])
            else:
                extra_meters[k].update(v)
            stats[k] = extra_meters[k].avg
        progress.log(stats, tag='train', step=stats['num_updates'])

        # ignore the first mini-batch in words-per-second calculation
        if i == 0:
            trainer.get_meter('wps').reset()

        num_updates = trainer.get_num_updates()
        if (
            not args.disable_validation
            and args.save_interval_updates > 0
            # and num_updates % args.save_interval_updates == 0
            and num_updates > 0
        ):
            valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets)
            checkpoint_utils.save_checkpoint(args, trainer, epoch_itr, valid_losses[0])

        if num_updates >= max_update:
            break

    # log end-of-epoch stats
    stats = get_training_stats(trainer)
    for k, meter in extra_meters.items():
        stats[k] = meter.avg
    progress.print(stats, tag='train', step=stats['num_updates'])

    # reset training meters
    for k in [
        'train_loss', 'train_nll_loss', 'wps', 'ups', 'wpb', 'bsz', 'gnorm', 'clip',
    ]:
        meter = trainer.get_meter(k)
        if meter is not None:
            meter.reset()


def get_training_stats(trainer):
    stats = collections.OrderedDict()
    stats['loss'] = trainer.get_meter('train_loss')
    if trainer.get_meter('train_nll_loss').count > 0:
        nll_loss = trainer.get_meter('train_nll_loss')
        stats['nll_loss'] = nll_loss
    else:
        nll_loss = trainer.get_meter('train_loss')
    stats['ppl'] = utils.get_perplexity(nll_loss.avg)
    stats['wps'] = trainer.get_meter('wps')
    stats['ups'] = trainer.get_meter('ups')
    stats['wpb'] = trainer.get_meter('wpb')
    stats['bsz'] = trainer.get_meter('bsz')
    stats['num_updates'] = trainer.get_num_updates()
    stats['lr'] = trainer.get_lr()
    stats['gnorm'] = trainer.get_meter('gnorm')
    stats['clip'] = trainer.get_meter('clip')
    stats['oom'] = trainer.get_meter('oom')
    if trainer.get_meter('loss_scale') is not None:
        stats['loss_scale'] = trainer.get_meter('loss_scale')
    stats['wall'] = round(trainer.get_meter('wall').elapsed_time)
    stats['train_wall'] = trainer.get_meter('train_wall')
    return stats


def validate(args, trainer, task, epoch_itr, subsets, xml_estimator=None, evaluate=False):
    """Evaluate the model on the validation set(s) and return the losses."""
    valid_losses = []
    for subset in subsets:
        # Initialize data iterator
        itr = task.get_batch_iterator(
            dataset=task.dataset(subset),
            max_tokens=args.max_tokens_valid,
            max_sentences=args.max_sentences_valid,
            max_positions=utils.resolve_max_positions(
                task.max_positions(),
                trainer.get_model().max_positions(),
            ),
            ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=args.required_batch_size_multiple,
            seed=args.seed,
            num_shards=args.distributed_world_size,
            shard_id=args.distributed_rank,
            num_workers=args.num_workers,
            sort=False,
        ).next_epoch_itr(shuffle=False)
        progress = progress_bar.build_progress_bar(
            args, itr, epoch_itr.epoch,
            prefix='valid on \'{}\' subset'.format(subset),
            no_progress_bar='simple'
        )

        # reset validation loss meters
        for k in ['valid_loss', 'valid_nll_loss']:
            meter = trainer.get_meter(k)
            if meter is not None:
                meter.reset()
        extra_meters = collections.defaultdict(lambda: AverageMeter())

        ter_prediction = []
        ter_gt = []
        ids = []
        fout = open(args.save_dir + '/' +subset + '_ce-qe-qsm-test.hter.order.xml', 'w')
        for sample in progress:
            if subset == 'train' and not evaluate and len(ter_prediction) > 1000:
                break
            log_output = trainer.valid_step(sample)

            for k, v in log_output.items():
                if k in ['loss', 'nll_loss', 'ntokens', 'nsentences', 'sample_size']:
                    continue
                elif k == 'posterior':
                    value = v.squeeze(-1).cpu().data
                    if isinstance(value.tolist(), float):
                        ter_prediction.append(value)
                        ter_gt.append(torch.Tensor(sample['ter']).squeeze(-1).data)
                    else:
                        ter_prediction.extend(value.tolist())
                        ter_gt.extend(torch.Tensor(sample['ter']).squeeze(-1).data)
                    ids.extend(sample['id'].tolist())
                    for v in value.tolist():
                        fout.write('%f\n' % v)
                    fout.flush()
                    continue
                extra_meters[k].update(v)
        if evaluate:
            splitor_index = args.restore_estimator_file.rfind('/')
            file_name = args.restore_estimator_file[:splitor_index]

            with open(args.save_dir + '/prediction', 'w', encoding='utf8') as fw:
                for id, t in zip(ids, ter_prediction):
                    fw.writelines(str(id) + " " + str(t) + '\n')

            sort_sorce(args.save_dir + '/prediction', args.save_dir + '/' +subset + '_ce-qe-qsm-test.hter.xml')
        pearson = np.corrcoef(ter_prediction, ter_gt)[0, 1]
        print('valid pearson: {}'.format(pearson))

        # log validation stats
        stats = get_valid_stats(trainer, args, extra_meters)
        for k, meter in extra_meters.items():
            stats[k] = meter.avg
        progress.print(stats, tag=subset, step=trainer.get_num_updates())
        valid_losses.append(pearson)
        valid_losses.append(
            stats[args.best_checkpoint_metric].avg
            if args.best_checkpoint_metric == 'loss'
            else stats[args.best_checkpoint_metric]
        )
    return valid_losses

def sort_sorce(filein, fileout):
    score_ids = {}
    with open(filein, 'r', encoding='utf8') as fr:
         for line in fr.readlines():
             line = line.strip('\n').split(' ')
             if len(line) == 2:
                 score_ids[int(line[0])] = line[1]

    score_ids = sorted(score_ids.items(),key=lambda item:item[0])
    with open(fileout, 'w', encoding='utf8') as fw:
        for id, score in score_ids:
            # print(id)
            fw.writelines(str(score) + '\n')

def get_valid_stats(trainer, args, extra_meters=None):
    stats = collections.OrderedDict()
    stats['loss'] = trainer.get_meter('valid_loss')
    if trainer.get_meter('valid_nll_loss').count > 0:
        nll_loss = trainer.get_meter('valid_nll_loss')
        stats['nll_loss'] = nll_loss
    else:
        nll_loss = stats['loss']
    stats['ppl'] = utils.get_perplexity(nll_loss.avg)
    stats['num_updates'] = trainer.get_num_updates()
    if hasattr(checkpoint_utils.save_checkpoint, 'best'):
        key = f'best_{args.best_checkpoint_metric}'
        best_function = max if args.maximize_best_checkpoint_metric else min

        current_metric = None
        if args.best_checkpoint_metric == 'loss':
            current_metric = stats['loss'].avg
        elif args.best_checkpoint_metric in extra_meters:
            current_metric = extra_meters[args.best_checkpoint_metric].avg
        elif args.best_checkpoint_metric in stats:
            current_metric = stats[args.best_checkpoint_metric]
        else:
            raise ValueError("best_checkpoint_metric not found in logs")

        stats[key] = best_function(
            checkpoint_utils.save_checkpoint.best,
            current_metric,
        )
    return stats


def distributed_main(i, args, start_rank=0):
    args.device_id = i
    if args.distributed_rank is None:  # torch.multiprocessing.spawn
        args.distributed_rank = start_rank + i
    main(args, init_distributed=True)


def cli_main():
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser)

    if args.distributed_init_method is None:
        distributed_utils.infer_init_method(args)

    if args.distributed_init_method is not None:
        # distributed training
        if torch.cuda.device_count() > 1 and not args.distributed_no_spawn:
            start_rank = args.distributed_rank
            args.distributed_rank = None  # assign automatically
            torch.multiprocessing.spawn(
                fn=distributed_main,
                args=(args, start_rank),
                nprocs=torch.cuda.device_count(),
            )
        else:
            distributed_main(args.device_id, args)
    elif args.distributed_world_size > 1:
        # fallback for single node with multiple GPUs
        assert args.distributed_world_size <= torch.cuda.device_count()
        port = random.randint(10000, 20000)
        args.distributed_init_method = 'tcp://localhost:{port}'.format(port=port)
        args.distributed_rank = None  # set based on device id
        if max(args.update_freq) > 1 and args.ddp_backend != 'no_c10d':
            print('| NOTE: you may get better performance with: --ddp-backend=no_c10d')
        torch.multiprocessing.spawn(
            fn=distributed_main,
            args=(args, ),
            nprocs=args.distributed_world_size,
        )
    else:
        # single GPU training
        main(args)


if __name__ == '__main__':
    cli_main()
