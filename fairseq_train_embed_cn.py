#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.







##############################################################################
##############################################################################
####
####   Added an Ranger optimizer usable in fairseq
####
##############################################################################
##############################################################################




import torch.optim
from fairseq.optim import FairseqOptimizer, register_optimizer

from ranger import Ranger

@register_optimizer('ranger')
class FairseqRanger(FairseqOptimizer):

    def __init__(self, args, params):
        super().__init__(args)
        self._optimizer = Ranger(params, **self.optimizer_config)

    @staticmethod
    def add_args(parser):
        """Add optimizer-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--adam-betas', default='(0.95, 0.999)', metavar='B',
                            help='betas for Adam optimizer')
        parser.add_argument('--adam-eps', type=float, default=1e-5, metavar='D',
                            help='epsilon for Adam optimizer')
        parser.add_argument('--weight-decay', '--wd', default=0.0, type=float, metavar='WD',
                            help='weight decay')
        # fmt: on

    @property
    def optimizer_config(self):
        """
        Return a kwarg dictionary that will be used to override optimizer
        args stored in checkpoints. This allows us to load a checkpoint and
        resume training using a different set of optimizer args, e.g., with a
        different learning rate.
        """
        return {
            'lr': self.args.lr[0],
            'betas': eval(self.args.adam_betas),
            'eps': self.args.adam_eps,
            'weight_decay': self.args.weight_decay,
        }




##############################################################################
##############################################################################
####
####   Added util to make decayed learning rates in layers of transformer
####
##############################################################################
##############################################################################




def get_decayed_param_groups(named_parameters, 
                             num_layers=None, 
                             lr=3e-5, 
                             lr_decay=0.75, #0.908517, 
                             weight_decay=None, 
                             freeze_transformer=False):
  lr_factors = []
  for k, v in named_parameters:
      if not v.requires_grad:
        continue
      if freeze_transformer and ('sentence_encoder.layers' in k or 'embed_tokens.weight' in k or 'embed_positions' in k):
        print('no grad:', k)
        v.requires_grad = False
        continue
      param = {
          'params': v,
      }
      if lr_decay and lr_decay != 1:
        factor = 1
        if 'sentence_encoder.layers' in k:
          layer = int(re.search(r'.layers.(\d+)',k).group(1))
          factor = lr_decay**(num_layers-layer)

        elif 'embed_tokens.weight' in k or 'embed_positions' in k:
          layer = 0
          factor = lr_decay**(num_layers-layer)

        param['lr'] = lr * factor
      if weight_decay and weight_decay != 0:
        param['weight_decay'] = 0.0 if 'layer_norm' in k or 'bias' in k else weight_decay
        
      lr_factors.append(param)
      print(param)
  return lr_factors
      
      



##############################################################################
##############################################################################
####
####   copy from pytorch/fairseq/train.py
####
##############################################################################
##############################################################################






"""
Train a new model on one or across multiple GPUs.
"""

import collections
import math
import random

import numpy as np
import torch

from fairseq import checkpoint_utils, distributed_utils, options, progress_bar, tasks, utils
from fairseq.data import iterators

from fairseq.meters import AverageMeter, StopwatchMeter

from collections import OrderedDict
import contextlib
from itertools import chain
import os
import sys

from fairseq import checkpoint_utils, distributed_utils, models, optim, utils
from fairseq.meters import AverageMeter, StopwatchMeter, TimeMeter
from fairseq.optim import lr_scheduler
import re
import types










class Trainer(object):
    """Main class for data parallel training.
    This class supports synchronous distributed data parallel training,
    where multiple workers each have a full model replica and gradients
    are accumulated across workers before each update. We use
    :class:`~torch.nn.parallel.DistributedDataParallel` to handle
    communication of the gradients across workers.
    """

    def __init__(self, args, task, model, criterion, dummy_batch=None, oom_batch=None):
        self.args = args
        self.task = task

        # copy model and criterion to current device
        self._criterion = criterion
        self._model = model
        self.cuda = torch.cuda.is_available() and not args.cpu
        if args.fp16:
            self._criterion = self._criterion.half()
            self._model = self._model.half()
        if self.cuda:
            self._criterion = self._criterion.cuda()
            self._model = self._model.cuda()

        self._dummy_batch = dummy_batch
        self._oom_batch = oom_batch or dummy_batch

        self._lr_scheduler = None
        self._num_updates = 0
        self._optim_history = None
        self._optimizer = None
        self._prev_grad_norm = None
        self._wrapped_criterion = None
        self._wrapped_model = None

        self.init_meters(args)

    def init_meters(self, args):
        self.meters = OrderedDict()
        self.meters['train_loss'] = AverageMeter()
        self.meters['train_nll_loss'] = AverageMeter()
        self.meters['valid_loss'] = AverageMeter()
        self.meters['valid_nll_loss'] = AverageMeter()
        self.meters['wps'] = TimeMeter()       # words per second
        self.meters['ups'] = TimeMeter()       # updates per second
        self.meters['wpb'] = AverageMeter()    # words per batch
        self.meters['bsz'] = AverageMeter()    # sentences per batch
        self.meters['gnorm'] = AverageMeter()  # gradient norm
        self.meters['clip'] = AverageMeter()   # % of updates clipped
        self.meters['oom'] = AverageMeter()    # out of memory
        if args.fp16:
            self.meters['loss_scale'] = AverageMeter()  # dynamic loss scale
        self.meters['wall'] = TimeMeter()      # wall time in seconds
        self.meters['train_wall'] = StopwatchMeter()  # train wall time in seconds

    @property
    def criterion(self):
        if self._wrapped_criterion is None:
            if (
                utils.has_parameters(self._criterion)
                and self.args.distributed_world_size > 1
                and not self.args.use_bmuf
            ):
                self._wrapped_criterion = models.DistributedFairseqModel(
                    self.args, self._criterion
                )
            else:
                self._wrapped_criterion = self._criterion
        return self._wrapped_criterion

    @property
    def model(self):
        if self._wrapped_model is None:
            if self.args.distributed_world_size > 1 and not self.args.use_bmuf:
                self._wrapped_model = models.DistributedFairseqModel(
                    self.args, self._model,
                )
            else:
                self._wrapped_model = self._model
        return self._wrapped_model

    @property
    def optimizer(self):
        if self._optimizer is None:
            self._build_optimizer()
        return self._optimizer

    @property
    def lr_scheduler(self):
        if self._lr_scheduler is None:
            self._build_optimizer()  # this will initialize self._lr_scheduler
        return self._lr_scheduler

    def _build_optimizer(self):

        ##############################################################################
        ##############################################################################
        ####
        ####   added decayed parameters if set, try later
        ####
        ##############################################################################
        ##############################################################################

        params = get_decayed_param_groups(chain(self.model.named_parameters(), self.criterion.named_parameters()), self.args.lr_decay_layers, lr_decay=self.args.lr_decay, freeze_transformer = self.args.freeze_transformer)
        '''
        params = list(
            filter(
                lambda p: p.requires_grad,
                chain(self.model.parameters(), self.criterion.parameters()),
            )
        )
        '''
        if self.args.fp16:
            if self.cuda and torch.cuda.get_device_capability(0)[0] < 7:
                print('| WARNING: your device does NOT support faster training with --fp16, '
                      'please switch to FP32 which is likely to be faster')
            if self.args.memory_efficient_fp16:
                self._optimizer = optim.MemoryEfficientFP16Optimizer.build_optimizer(self.args, params)
            else:
                self._optimizer = optim.FP16Optimizer.build_optimizer(self.args, params)
        else:
            if self.cuda and torch.cuda.get_device_capability(0)[0] >= 7:
                print('| NOTICE: your device may support faster training with --fp16')
            self._optimizer = optim.build_optimizer(self.args, params)

        if self.args.use_bmuf:
            self._optimizer = optim.FairseqBMUF(self.args, self._optimizer)

        # We should initialize the learning rate scheduler immediately after
        # building the optimizer, so that the initial learning rate is set.
        self._lr_scheduler = lr_scheduler.build_lr_scheduler(self.args, self.optimizer)
        self._lr_scheduler.step_update(0)

    def save_checkpoint(self, filename, extra_state):
        """Save all training state in a checkpoint file."""
        if distributed_utils.is_master(self.args):  # only save one checkpoint
            extra_state['train_meters'] = self.meters
            checkpoint_utils.save_state(
                filename, self.args, self.get_model().state_dict(), self.get_criterion(),
                self.optimizer, self.lr_scheduler, self.get_num_updates(),
                self._optim_history, extra_state,
            )

    def load_checkpoint(
        self,
        filename,
        reset_optimizer=False,
        reset_lr_scheduler=False,
        optimizer_overrides=None,
        reset_meters=False,
    ):
        """Load all training state from a checkpoint file."""
        extra_state, self._optim_history, last_optim_state = None, [], None

        if os.path.exists(filename):
            state = checkpoint_utils.load_checkpoint_to_cpu(filename)

            # load model parameters
            try:
                self.get_model().load_state_dict(state['model'], strict=True)
                if utils.has_parameters(self.get_criterion()):
                    self.get_criterion().load_state_dict(state['criterion'], strict=True)
            except Exception as e:

                ##############################################################################
                ##############################################################################
                ####
                ####   Lazy Hack... 
                ####
                ##############################################################################
                ##############################################################################

                print(
                    e, 
                    'Cannot load model parameters from checkpoint {}; '
                    'please ensure that the architectures match.'.format(filename)
                )
                try:
                    print('!!! Training Continued Ignoring The Above Error !!!')
                    self.get_model().load_state_dict(state['model'], strict=False)
                    if utils.has_parameters(self.get_criterion()):
                        self.get_criterion().load_state_dict(state['criterion'], strict=False)
                except:
                    raise Exception('version is %s'%(sys.version_info, ))

            extra_state = state['extra_state']
            self._optim_history = state['optimizer_history']
            last_optim_state = state.get('last_optimizer_state', None)

        if last_optim_state is not None and not reset_optimizer:
            # rebuild optimizer after loading model, since params may have changed
            self._build_optimizer()

            # only reload optimizer and lr_scheduler if they match
            last_optim = self._optim_history[-1]
            assert last_optim['criterion_name'] == self.get_criterion().__class__.__name__, \
                'Criterion does not match; please reset the optimizer (--reset-optimizer).'
            assert last_optim['optimizer_name'] == self.optimizer.__class__.__name__, \
                'Optimizer does not match; please reset the optimizer (--reset-optimizer).'

            if not reset_lr_scheduler:
                self.lr_scheduler.load_state_dict(last_optim['lr_scheduler_state'])
            self.optimizer.load_state_dict(last_optim_state, optimizer_overrides)

            self.set_num_updates(last_optim['num_updates'])

        if extra_state is not None:
            epoch = extra_state['train_iterator']['epoch']
            print('| loaded checkpoint {} (epoch {} @ {} updates)'.format(
                filename, epoch, self.get_num_updates()))

            self.lr_step(epoch)

            if 'train_meters' in extra_state and not reset_meters:
                self.meters.update(extra_state['train_meters'])
                del extra_state['train_meters']

                # reset TimeMeters, since their start times don't make sense anymore
                for meter in self.meters.values():
                    if isinstance(meter, TimeMeter):
                        meter.reset()
        else:
            print('| no existing checkpoint found {}'.format(filename))

        return extra_state

    def get_train_iterator(self, epoch, combine=True):
        """Return an EpochBatchIterator over the training set for a given epoch."""
        print('| loading train data for epoch {}'.format(epoch))
        self.task.load_dataset(self.args.train_subset, epoch=epoch, combine=combine)
        return self.task.get_batch_iterator(
            dataset=self.task.dataset(self.args.train_subset),
            max_tokens=self.args.max_tokens,
            max_sentences=self.args.max_sentences,
            max_positions=utils.resolve_max_positions(
                self.task.max_positions(),
                self.model.max_positions(),
            ),
            ignore_invalid_inputs=True,
            required_batch_size_multiple=self.args.required_batch_size_multiple,
            seed=self.args.seed,
            num_shards=self.args.distributed_world_size,
            shard_id=self.args.distributed_rank,
            num_workers=self.args.num_workers,
            epoch=epoch,
        )

    def train_step(self, samples, dummy_batch=False, raise_oom=False):
        """Do forward, backward and parameter update."""
        if self._dummy_batch is None:
            self._dummy_batch = samples[0]

        self._set_seed()
        self.model.train()
        self.criterion.train()
        self.zero_grad()

        if not dummy_batch:
            self.meters['train_wall'].start()

        # forward and backward pass
        logging_outputs, sample_sizes, ooms = [], [], 0
        for i, sample in enumerate(samples):
            sample = self._prepare_sample(sample)
            if sample is None:
                # when sample is None, run forward/backward on a dummy batch
                # and ignore the resulting gradients
                sample = self._prepare_sample(self._dummy_batch)
                ignore_grad = True
            else:
                ignore_grad = False

            def maybe_no_sync():
                """
                Whenever *samples* contains more than one mini-batch, we
                want to accumulate gradients locally and only call
                all-reduce in the last backwards pass.
                """
                if (
                    self.args.distributed_world_size > 1
                    and hasattr(self.model, 'no_sync')
                    and i < len(samples) - 1
                ):
                    return self.model.no_sync()
                else:
                    return contextlib.ExitStack()  # dummy contextmanager

            try:
                with maybe_no_sync():
                    # forward and backward
                    loss, sample_size, logging_output = self.task.train_step(
                        sample, self.model, self.criterion, self.optimizer,
                        ignore_grad
                    )

                if not ignore_grad:
                    logging_outputs.append(logging_output)
                    sample_sizes.append(sample_size)
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    msg = (
                        '| WARNING: ran out of memory with exception: '
                        + '{};'.format(e)
                        + '\n Skipping batch'
                    )
                    # TODO: print should really go to logger, this print goes
                    # to stdout, which is buffered, which in many case is not
                    # printed out if another exception happens
                    # print(msg)
                    print(msg, file=sys.stderr)
                    if raise_oom:
                        raise ValueError(msg)
                    ooms += 1
                    self.zero_grad()
                else:
                    raise e

        if ooms > 0 and self._oom_batch is not None:
            self.handle_ooms(ooms)

        if dummy_batch:
            return None

        # gather logging outputs from all replicas
        if self.args.distributed_world_size > 1 and (
            (not self.args.use_bmuf)
            or (
                self.args.use_bmuf
                and (self.get_num_updates() + 1) % self.args.global_sync_iter == 0
            )
        ):
            logging_outputs, sample_sizes, ooms, prev_norms = \
                zip(*distributed_utils.all_gather_list(
                    [logging_outputs, sample_sizes, ooms, self._prev_grad_norm],
                ))
            logging_outputs = list(chain.from_iterable(logging_outputs))
            sample_sizes = list(chain.from_iterable(sample_sizes))
            ooms = sum(ooms)

            if not self.args.use_bmuf:
                assert (
                    all(norm == prev_norms[0] for norm in prev_norms)
                    or all(math.isnan(norm) or math.isinf(norm) for norm in prev_norms)
                ), 'Fatal error: gradients are inconsistent between workers'

        self.meters['oom'].update(ooms, len(samples))
        if ooms == self.args.distributed_world_size * len(samples):
            print('| WARNING: OOM in all workers, skipping update')
            self.zero_grad()
            return None

        # aggregate logging outputs and sample sizes
        logging_output = self.task.aggregate_logging_outputs(
            logging_outputs, self.get_criterion()
        )
        sample_size = self.task.grad_denom(sample_sizes, self.get_criterion())

        if not all(k in logging_output for k in ['ntokens', 'nsentences']):
            raise Exception((
                'Please update the {}.aggregate_logging_outputs() method to '
                'return ntokens and nsentences'
            ).format(self.task.__class__.__name__))

        try:
            # normalize grads by sample size
            if sample_size > 0:
                self.optimizer.multiply_grads(self.args.distributed_world_size / float(sample_size))

            # clip grads
            grad_norm = self.optimizer.clip_grad_norm(self.args.clip_norm)
            self._prev_grad_norm = grad_norm

            # take an optimization step
            self.optimizer.step()
            self.set_num_updates(self.get_num_updates() + 1)

            # task specific update per step
            self.task.update_step(self._num_updates)

            # update meters
            ntokens = logging_output.get('ntokens', 0)
            nsentences = logging_output.get('nsentences', 0)
            self.meters['wps'].update(ntokens)
            self.meters['ups'].update(1.)
            self.meters['wpb'].update(ntokens)
            self.meters['bsz'].update(nsentences)
            self.meters['gnorm'].update(grad_norm)
            self.meters['clip'].update(
                1. if grad_norm > self.args.clip_norm and self.args.clip_norm > 0 else 0.
            )
            self.meters['train_loss'].update(logging_output.get('loss', 0), sample_size)
            if 'train_acc' in self.meters:
                self.meters['train_acc'].update(
                    logging_output.get('acc', 0), sample_size)

            if 'nll_loss' in logging_output:
                self.meters['train_nll_loss'].update(logging_output.get('nll_loss', 0), ntokens)
        except OverflowError as e:
            print('| WARNING: overflow detected, ' + str(e))
            self.zero_grad()
            logging_output = None

        if self.args.fp16:
            self.meters['loss_scale'].reset()
            self.meters['loss_scale'].update(self.optimizer.scaler.loss_scale)

        self.meters['train_wall'].stop()

        return logging_output

    def valid_step(self, sample, raise_oom=False):
        """Do forward pass in evaluation mode."""
        with torch.no_grad():
            self.model.eval()
            self.criterion.eval()

            sample = self._prepare_sample(sample)
            if sample is None:
                sample = self._prepare_sample(self._dummy_batch)
                ignore_results = True
            else:
                ignore_results = False

            try:
                _loss, sample_size, logging_output = self.task.valid_step(
                    sample, self.model, self.criterion
                )
            except RuntimeError as e:
                if 'out of memory' in str(e) and not raise_oom:
                    print('| WARNING: ran out of memory, retrying batch')
                    for p in self.model.parameters():
                        if p.grad is not None:
                            p.grad = None  # free some memory
                    if self.cuda:
                        torch.cuda.empty_cache()
                    return self.valid_step(sample, raise_oom=True)
                else:
                    raise e

            if ignore_results:
                logging_output, sample_size = {}, 0

        # gather logging outputs from all replicas
        if self.args.distributed_world_size > 1:
            logging_output, sample_size = zip(*distributed_utils.all_gather_list(
                [logging_output, sample_size],
            ))
            logging_output = list(logging_output)
            sample_size = list(sample_size)
        else:
            logging_output = [logging_output]
            sample_size = [sample_size]

        # aggregate logging outputs and sample sizes
        logging_output = self.task.aggregate_logging_outputs(
            logging_output, self.get_criterion()
        )
        sample_size = self.task.grad_denom(
            sample_size, self.get_criterion()
        )

        # update meters for validation
        ntokens = logging_output.get('ntokens', 0)
        self.meters['valid_loss'].update(logging_output.get('loss', 0), sample_size)
        if 'valid_acc' in self.meters:
            self.meters['valid_acc'].update(
                logging_output.get('acc', 0), sample_size)

        if 'nll_loss' in logging_output:
            self.meters['valid_nll_loss'].update(logging_output.get('nll_loss', 0), ntokens)

        return logging_output

    def dummy_train_step(self, dummy_batch):
        """Dummy training step for warming caching allocator."""
        self.train_step(dummy_batch, dummy_batch=True)
        self.zero_grad()

    def handle_ooms(self, number_of_ooms):
        """
        c10d accumulates/syncs gradients between gpus during backward pass.
        In case of OOMs, gpus may fail to sync, so we manually iterate
        extra to make sure each gpu makes same number of iterations.
        """
        for _ in range(number_of_ooms):
            self.train_step([self._oom_batch], True)

    def zero_grad(self):
        self.optimizer.zero_grad()

    def lr_step(self, epoch, val_loss=None):
        """Adjust the learning rate based on the validation loss."""
        self.lr_scheduler.step(epoch, val_loss)
        # prefer updating the LR based on the number of steps
        return self.lr_step_update()

    def lr_step_update(self):
        """Update the learning rate after each update."""
        return self.lr_scheduler.step_update(self.get_num_updates())

    def get_lr(self):
        """Get the current learning rate."""
        return self.optimizer.get_lr()

    def get_model(self):
        """Get the (non-wrapped) model instance."""
        return self._model

    def get_criterion(self):
        """Get the (non-wrapped) criterion instance."""
        return self._criterion

    def get_meter(self, name):
        """Get a specific meter by name."""
        if name not in self.meters:
            return None
        return self.meters[name]

    def get_num_updates(self):
        """Get the number of parameters updates."""
        return self._num_updates

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        self._num_updates = num_updates
        self.lr_step_update()

    def _prepare_sample(self, sample):
        if sample is None or len(sample) == 0:
            return None

        if self.cuda:
            sample = utils.move_to_cuda(sample)

        def apply_half(t):
            if t.dtype is torch.float32:
                return t.half()
            return t

        if self.args.fp16:
            sample = utils.apply_to_sample(apply_half, sample)

        return sample

    def _set_seed(self):
        # Set seed based on args.seed and the update number so that we get
        # reproducible results when resuming from checkpoints
        seed = self.args.seed + self.get_num_updates()
        torch.manual_seed(seed)
        if self.cuda:
            torch.cuda.manual_seed(seed)

def main(args, init_distributed=False):
    utils.import_user_module(args)

    assert args.max_tokens is not None or args.max_sentences is not None, \
        'Must specify batch size either with --max-tokens or --max-sentences'

    # Initialize CUDA and distributed training
    if torch.cuda.is_available() and not args.cpu:
        torch.cuda.set_device(args.device_id)
    np.random.seed(args.seed)
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
        task.load_dataset(valid_sub_split, combine=False, epoch=0)

    # Build model and criterion
    model = task.build_model(args)
    criterion = task.build_criterion(args)
    print(model)
    print('| model {}, criterion {}'.format(args.arch, criterion.__class__.__name__))
    print('| num. model params: {} (num. trained: {})'.format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    ))

    # Build trainer
    trainer = Trainer(args, task, model, criterion)
    print('| training on {} GPUs'.format(args.distributed_world_size))
    print('| max tokens per GPU = {} and max sentences per GPU = {}'.format(
        args.max_tokens,
        args.max_sentences,
    ))

    # Load the latest checkpoint if one is available and restore the
    # corresponding train iterator
    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(args, trainer)

    # Train until the learning rate gets too small
    max_epoch = args.max_epoch or math.inf
    max_update = args.max_update or math.inf
    lr = trainer.get_lr()
    train_meter = StopwatchMeter()
    train_meter.start()
    valid_subsets = args.valid_subset.split(',')
    while lr > args.min_lr and epoch_itr.epoch < max_epoch and trainer.get_num_updates() < max_update:
        # train for one epoch
        train(args, trainer, task, epoch_itr)

        if not args.disable_validation and epoch_itr.epoch % args.validate_interval == 0:
            valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets)
        else:
            valid_losses = [None]

        # only use first validation loss to update the learning rate
        lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])

        # save checkpoint
        if epoch_itr.epoch % args.save_interval == 0:
            checkpoint_utils.save_checkpoint(args, trainer, epoch_itr, valid_losses[0])

        if ':' in getattr(args, 'data', ''):
            # sharded data: get train iterator for next epoch
            epoch_itr = trainer.get_train_iterator(epoch_itr.epoch)
    train_meter.stop()
    print('| done training in {:.1f} seconds'.format(train_meter.sum))


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
            and num_updates % args.save_interval_updates == 0
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


def validate(args, trainer, task, epoch_itr, subsets):
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

        for sample in progress:
            log_output = trainer.valid_step(sample)

            for k, v in log_output.items():
                if k in ['loss', 'nll_loss', 'ntokens', 'nsentences', 'sample_size']:
                    continue
                extra_meters[k].update(v)

        # log validation stats
        stats = get_valid_stats(trainer, args, extra_meters)
        for k, meter in extra_meters.items():
            stats[k] = meter.avg
        progress.print(stats, tag=subset, step=trainer.get_num_updates())

        valid_losses.append(
            stats[args.best_checkpoint_metric].avg
            if args.best_checkpoint_metric == 'loss'
            else stats[args.best_checkpoint_metric]
        )
    return valid_losses


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
        key = 'best_{0}'.format(args.best_checkpoint_metric)
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
##############################################################################
##############################################################################
####
####   Added an argument
####
##############################################################################
##############################################################################
    parser.add_argument('--lr_decay', default=1, type=float, 
                        help='Learning rate decay factor, 1.0 = no decay')
    parser.add_argument('--lr_decay_layers', default=24, type=int, 
                        help='Number of layers for learning rate decay')
    parser.add_argument('--freeze_transformer', dest='freeze_transformer', action='store_true',
                        help='Whether to freeze the weights in transformer')
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


##############################################################################
##############################################################################
####
####   copy from fairseq... for tasks, criterion, and architectures
####
##############################################################################
##############################################################################


import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils


from fairseq.tasks import FairseqTask, register_task
from fairseq.criterions import FairseqCriterion, register_criterion

from fairseq.models import (
    FairseqDecoder,
    FairseqLanguageModel,
    register_model,
    register_model_architecture,
)

from fairseq.data import (
    data_utils,
    Dictionary,
    encoders,
    IdDataset,
    ListDataset,
    NestedDictionaryDataset,
    NumSamplesDataset,
    NumelDataset,
    PadDataset,
    SortDataset,
)
from fairseq.modules import (
    LayerNorm,
    TransformerSentenceEncoder,
)
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.models.roberta.hub_interface import RobertaHubInterface

import argparse
from fairseq.data import Dictionary
from fairseq.optim.fp16_optimizer import MemoryEfficientFP16Optimizer
from glob import glob
import numpy as np
from torch.nn import CrossEntropyLoss
import json
from copy import deepcopy
from time import time
from multiprocessing import Pool
import multiprocessing
import gc
import random
from tqdm import tqdm
import os


# specially made for roberta
from tokenizer.roberta import RobertaTokenizer, MASKED, NOT_MASKED, IS_MAX_CONTEXT, NOT_IS_MAX_CONTEXT
from tokenizer.validate import validate


roberta_directory = './roberta.large'


max_seq_length   = 192
max_query_length = 128
doc_stride       = 128

default_choices = []
get_tokenizer = lambda: RobertaTokenizer(config_dir=roberta_directory)

tk = tokenizer =  get_tokenizer()


#Data Utilities

import marshal
def read(dat):
    a, b = marshal.loads(dat)
    a = np.frombuffer(a, dtype=np.uint16).astype(np.int32)
    b = np.frombuffer(b, dtype=np.uint16).astype(np.int32)
    return a, b

def fread(f):
    a, b = marshal.load(f)
    #a = np.frombuffer(a, dtype=np.uint16).astype(np.int32)
    #b = np.frombuffer(b, dtype=np.uint16).astype(np.int32)
    return a, b
            

def pad(list_of_tokens, 
        dtype=np.long,
        torch_tensor=None,
        max_seq_length=max_seq_length,
        pad_idx=1):
    k = np.empty((len(list_of_tokens),max_seq_length), dtype=dtype)
    k.fill(pad_idx)
    i = 0
    for tokens in list_of_tokens:
        k[i,:len(tokens)] = tokens
        i += 1
    return k if torch_tensor is None else torch_tensor(k)

from torch.utils.data.dataset import Dataset


def chunks(l, n):
    if type(l) == type((e for e in range(1))):
        it = iter(l)
        while True:
            out = []
            try:
                for _ in range(n):
                    out.append(next(it))
            except StopIteration:
                yield out
                break

            yield out
    else:
    
        for i in range(0, len(l), n):
            yield l[i:i + n]

def from_records(records):
    """
    Args:
        records (string): Path to the csv file with annotations.
    """
  
    fn_style = isinstance(records,str)
    if fn_style:
      def from_file(fn):
        with open(fn, 'rb') as f:
            while True:
                try:
                    record = fread(f)
                    yield record
                except EOFError:
                    break
      records = from_file(records)

    records = list(records)
      
    prepared_records = []
    for record_samples in chunks(records,48):
        a, b = zip(*record_samples) if fn_style else zip(*(read(record) for record in record_samples))
        #a = pad(a,dtype=np.long, torch_tensor=torch.LongTensor)
        #b = pad(b,dtype=np.long, torch_tensor=torch.LongTensor)

        for e in zip(a,b):
            yield e






from fairseq import utils

class FnnLayer(nn.Module):
    def __init__(self, hidden_size, activation_fn='gelu', dropout=0.2):
        super(FnnLayer, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, hidden_states):
        return self.dropout(self.activation(self.dense(hidden_states))) + hidden_states


from fairseq.models import BaseFairseqModel
import transformers


class BertForQAEmbed(transformers.BertPreTrainedModel):
    def __init__(self, config):
        super(BertForQAEmbed, self).__init__(config)

        self.bert = transformers.BertModel(config)
        self.cls = transformers.modeling_bert.BertOnlyNSPHead(config)

        self.q_fnn_layer = FnnLayer(config.hidden_size)
        self.a_fnn_layer = FnnLayer(config.hidden_size)

        self.init_weights()
'''
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None,
                next_sentence_label=None):

        outputs = 

        pooled_output = outputs[1]

        seq_relationship_score = self.cls(pooled_output)

        outputs = (seq_relationship_score,) + outputs[2:]  # add hidden states and attention if they are here
        if next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            outputs = (next_sentence_loss,) + outputs

        return outputs  # (next_sentence_loss), seq_relationship_score, (hidden_states), (attentions)
'''
    def get_pooled_output_average_tokens_from_last_layer(self, x):
        y = self.bert(x)[0]
        return ( y * (1 - x.eq(0).unsqueeze(-1).type_as(y)) ).mean(1)

    def forward(self, q=None, a=None, return_loss=False, **kwargs):
        has_q = q is not None
        has_a = a is not None
        if not has_q and not has_a:
          raise Exception('??')
        batch_size = q.size(0) if has_q else a.size(0)

        if has_q:
          q_embed = self.q_fnn_layer(self.get_pooled_output_average_tokens_from_last_layer(q))   # if average all tokens, needs : * 
          #q_embed = q_embed / q_embed.norm(dim=1)[:,None]
        if has_a:
          a_embed = self.a_fnn_layer(self.get_pooled_output_average_tokens_from_last_layer(a))
          #a_embed = a_embed / a_embed.norm(dim=1)[:,None]

        outputs = () 

        if return_loss:
            if not (has_q and has_a):
              raise Exception('Cannot calculate loss without both q and a')
            
            similarity_matrix = torch.mm(q_embed,a_embed.t())
            
            targets = torch.arange(batch_size).cuda()   
            
            loss = torch.nn.functional.cross_entropy(similarity_matrix, targets)
            
            corrects = targets.eq(torch.argmax(similarity_matrix, axis=1)).sum()
            targets.detach()
            outputs = (loss,corrects)

        else:
            if has_q:
              outputs = (q_embed,)
              
            if has_a:
              outputs = outputs + (a_embed,)
              
        return outputs


@register_model('roberta_qa_embed_hf')
class RobertaQAEmbedodelHF(transformers.BertForQAEmbed, BaseFairseqModel):

    def __init__(self, *args, **kwargs):
        transformers.BertForQAEmbed.__init__(self, *args, **kwargs)



    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        return RobertaQAEmbedodelHF(transformers.modeling_bert.BertConfig.from_json_file('/'.join(args.restore_file.split('/')[:-1]) + '/' + "config.json")	)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--encoder-layers', type=int, metavar='L',
                            help='num encoder layers')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='H',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='F',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='A',
                            help='num encoder attention heads')
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--pooler-activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use for pooler layer')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN')
        parser.add_argument('--pooler-dropout', type=float, metavar='D',
                            help='dropout probability in the masked_lm pooler layers')
        parser.add_argument('--max-positions', type=int,
                            help='number of positional embeddings to learn')
        parser.add_argument('--load-checkpoint-heads', action='store_true',
                            help='(re-)register and load heads when loading checkpoints')




@register_model_architecture('roberta_qa_embed_hf', 'roberta_qa_embed_hf_large')
def roberta_large_architecture(args):
    pass







from fairseq.data import BaseWrapperDataset
@register_task('qa_embed_hf')
class QAEmbedTask(FairseqTask):
    """Task for training masked language models (e.g., BERT, RoBERTa)."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument('data', help='colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner')
        parser.add_argument('--default_choices', default='', type=str)
        #max_positions

    def __init__(self, args):
        super().__init__(args)

    @classmethod
    def setup_task(cls, args, **kwargs):
        cls.dictionary = DummyDictionary()
        #dictionary = cls.dictionary = Dictionary.load(os.path.join(os.path.dirname(args.restore_file), 'dict.txt'))
        #dictionary.add_symbol('<mask>')
        #print('| dictionary: {} types'.format(len(dictionary)))
        return cls(args)

    def load_dataset(self, split, epoch=0, combine=False):
        """Load a given dataset split.
        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        path = self.args.data

        questions = []
        answers = []
        
        lengths = []
        
        for q, a in tqdm(from_records(path)):
            questions.append(q)
            answers.append(a)
            lengths.append(max_seq_length)
            
        
        questions = BaseWrapperDataset(questions)
        answers = BaseWrapperDataset(answers)
        lengths = np.array(lengths, dtype=np.long)

        print('| loaded {} batches [size:{}] from: {}'.format(len(lengths), max_seq_length, path))

        shuffle = np.arange(len(lengths))

        self.datasets[split] = SortDataset(
            NestedDictionaryDataset(
                {
                    'id': IdDataset(),
                    'questions': questions,
                    'answers': answers,
                    'nsentences': NumSamplesDataset(),
                    'ntokens': NumelDataset(questions, reduce=True),
                },
                sizes=[lengths],
            ),
            sort_order=[
                shuffle,
            ],
        )

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary


@register_criterion('qa_embed')
class QAEmbedCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        if self.args.save_predictions is not None:
            self.prediction_h = open(self.args.save_predictions, 'w')
        else:
            self.prediction_h = None

    def __del__(self):
        if self.prediction_h is not None:
            self.prediction_h.close()

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--save-predictions', metavar='FILE',
                            help='file to save predictions to')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        # compute loss and accuracy
        questions = [np.frombuffer(e, dtype=np.uint16).astype(np.int32) for e in sample['questions']]
        answers = [np.frombuffer(e, dtype=np.uint16).astype(np.int32) for e in sample['answers']]
        
        questions = pad(questions,dtype=np.long, torch_tensor=torch.LongTensor, max_seq_length=max(len(e) for e in questions)).cuda()
        answers   = pad(answers,dtype=np.long, torch_tensor=torch.LongTensor, max_seq_length=max(len(e) for e in answers)).cuda()
        
        (loss, corrects) = model(questions, answers, return_loss=True)

        
        sample_size = questions.size(0) 
        questions.detach()
        answers.detach()
        
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'corrects': utils.item(corrects.data) if reduce else corrects.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['nsentences'],
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        corrects = sum(log.get('corrects', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        agg_output = {
            'loss': loss_sum / sample_size / math.log(2),
            'accuracy': corrects / sample_size,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }

        return agg_output
        
        
if __name__ == '__main__':
  cli_main()