# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
import types

import torch
import torch.optim

from fairseq.optim import FairseqOptimizer, register_optimizer

from tensorboardX import SummaryWriter
# writer = SummaryWriter(logdir='./log/ada/')
# # writer = SummaryWriter(logdir='./log/wmt/')

iter_idx = 0

@register_optimizer('adam2')
class FairseqAdam2(FairseqOptimizer):

    def __init__(self, args, params):
        super().__init__(args, params)

        self._optimizer = Adam2(params, **self.optimizer_config)
        self._optimizer.name = args.tb_tag + '_' + self._optimizer.name

    @staticmethod
    def add_args(parser):
        """Add optimizer-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--adam-betas', default='(0.9, 0.999)', metavar='B',
                            help='betas for Adam optimizer')
        parser.add_argument('--adam-eps', type=float, default=1e-8, metavar='D',
                            help='epsilon for Adam optimizer')
        parser.add_argument('--weight-decay', '--wd', default=0.0, type=float, metavar='WD',
                            help='weight decay')
        parser.add_argument('--tb-tag', default="", type=str,
                            help='tb tag')
        parser.add_argument('--amsgrad', action='store_true')
        parser.add_argument('--adam-freeze', default=5000, type=float)
        parser.add_argument('--adam-no-correction1', action='store_true')
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
            'amsgrad': self.args.amsgrad,
            'adam_freeze': self.args.adam_freeze,
            'adam_no_correction1': self.args.adam_no_correction1,
        }


class Adam2(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, adam_freeze=5000, adam_no_correction1=False):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad, adam_freeze=adam_freeze, adam_no_correction1=adam_no_correction1)
        self.name = '{}_{}_{}'.format(lr, betas[0], betas[1])
        super(Adam2, self).__init__(params, defaults)

    @property
    def supports_memory_efficient_fp16(self):
        return True

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        global iter_idx
        iter_idx += 1
        grad_list = list()
        mom_list = list()
        mom_2rd_list = list()

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            # if 'adam_1k' in self.name:
            #     writer_iter = iter_idx - group['adam_freeze']
            # else:
            #     writer_iter = iter_idx

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                    if amsgrad:
                        state['max_exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)
                    if amsgrad:
                        state['max_exp_avg_sq'] = state['max_exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                exp_avg_sq.mul_(beta2).addcmul_(1-beta2, grad, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                if group['adam_no_correction1']:
                    bias_correction1 = 1
                else:
                    bias_correction1 = (1 - beta1 ** state['step'])

                bias_correction2 = (1 - beta2 ** state['step'])**0.5
                step_size = group['lr'] * bias_correction2 / bias_correction1


                if 'adam_1k' not in self.name or state['step'] > group['adam_freeze']:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    exp_avg.mul_(beta1).add_(1 - beta1, grad)
                    p_data_fp32.addcdiv_(-step_size, exp_avg, denom)
                    p.data.copy_(p_data_fp32)

        #         if writer_iter > 0 and writer_iter % 300 == 0 or writer_iter in [1, 5, 10, 25, 50, 75, 100, 150, 200]:
        #             grad_list.extend( grad.abs().add_(1e-9).log().view(-1).tolist()  )
        #             mom_list.extend( exp_avg.abs().add_(1e-9).log().view(-1).tolist() )
        #             mom_2rd_list.extend( exp_avg_sq.abs().add_(1e-9).log().view(-1).tolist() )

        # if writer_iter > 0 and writer_iter % 300 == 0 or writer_iter in [1, 5, 10, 25, 50, 75, 100, 150, 200]:
        #     writer.add_histogram('grad/{}'.format(self.name), grad_list, writer_iter)
        #     writer.add_histogram('mom/{}'.format(self.name), mom_list, writer_iter)
        #     writer.add_histogram('mom_sq/{}'.format(self.name), mom_2rd_list, writer_iter)

        return loss
