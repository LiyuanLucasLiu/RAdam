# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from fairseq.optim.lr_scheduler import FairseqLRScheduler, register_lr_scheduler


@register_lr_scheduler('poly')
class PolySchedule(FairseqLRScheduler):
    """Decay the LR based on the inverse square root of the update number.

    We also support a warmup phase where we linearly increase the learning rate
    from some initial learning rate (``--warmup-init-lr``) until the configured
    learning rate (``--lr``). Thereafter we decay proportional to the number of
    updates, with a decay factor set to align with the configured learning rate.

    During warmup::

      lrs = torch.linspace(args.warmup_init_lr, args.lr, args.warmup_updates)
      lr = lrs[update_num]

    After warmup::

      decay_factor = args.lr * sqrt(args.warmup_updates)
      lr = decay_factor / sqrt(update_num)
    """

    def __init__(self, args, optimizer):
        super().__init__(args, optimizer)
        if len(args.lr) > 1:
            raise ValueError(
                'Cannot use a fixed learning rate schedule with inverse_sqrt.'
                ' Consider --lr-scheduler=fixed instead.'
            )

        # then, decay prop. to the inverse square root of the update number
        # self.warmup_end_lr = warmup_end_lr * args.warmup_updates**0.5
        self.min_lr = args.min_lr

        # initial learning rate
        self.lr = args.lr[0]
        self.optimizer.set_lr(self.lr)

        self.max_update = args.max_update

    @staticmethod
    def add_args(parser):
        """Add arguments to the parser for this LR scheduler."""
        # fmt: off
        parser.add_argument('--poly-pow', default=2, type=float, metavar='N',
                            help='ploy power')

    def step(self, epoch, val_loss=None):
        """Update the learning rate at the end of the given epoch."""
        super().step(epoch, val_loss)
        # we don't change the learning rate at epoch boundaries
        return self.optimizer.get_lr()

    def step_update(self, num_updates):
        """Update the learning rate after each update."""
        self.lr = (self.args.lr[0] - self.min_lr)* (1 - num_updates / self.max_update)**self.args.poly_pow + self.min_lr
        self.optimizer.set_lr(self.lr)
        return self.lr
