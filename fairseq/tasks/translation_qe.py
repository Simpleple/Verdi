import contextlib
import torch.nn.functional as F
import torch
import torch.nn as nn

from fairseq import modules, utils
from fairseq.tasks import register_task
from fairseq.mtqe.translation_m import TranslationMTask
from fairseq.mtqe.estimator import Estimator

import numpy as np

@contextlib.contextmanager
def eval(model):
    is_training = model.training
    model.eval()
    yield
    model.train(is_training)


@register_task('translation_qe')
class TranslationQETask(TranslationMTask):
    """
    Translation task for Mixture of Experts (MoE) models.

    See `"Mixture Models for Diverse Machine Translation: Tricks of the Trade"
    (Shen et al., 2019) <https://arxiv.org/abs/1902.07816>`_.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        TranslationMTask.add_args(parser)
        parser.add_argument('--method', default='hMoEup',
                            choices=['sMoElp', 'sMoEup', 'hMoElp', 'hMoEup'])
        parser.add_argument('--num-experts', default=3, type=int, metavar='N',
                            help='number of experts')
        parser.add_argument('--mean-pool-gating-network', action='store_true',
                            help='use a simple mean-pooling gating network')
        parser.add_argument('--mean-pool-gating-network-dropout', type=float,
                            help='dropout for mean-pooling gating network')
        parser.add_argument('--mean-pool-gating-network-encoder-dim', type=float,
                            help='encoder output dim for mean-pooling gating network')
        parser.add_argument('--gen-expert', type=int, default=0,
                            help='which expert to use for generation')

        parser.add_argument('--estimator-xml-dim', type=int, default=0,
                            help='use xml pretrained model as predictor')
        parser.add_argument('--estimator-transformer-dim', type=int, default=5140,
                            help='use transformer pretrained model as predictor')
        parser.add_argument('--share-estimator', type=int, default=0,
                            help='different predicotors share the same estimator')
        parser.add_argument('--estimator-xml-only', type=int, default=0,
                            help='only use xml pretrained model as predictor')
        parser.add_argument('--evaluate', type=int, default=0,
                            help='evaluate the restored model')
        parser.add_argument('--share-xml-dict', type=int, default=0,
                            help='evaluate the restored model')
        parser.add_argument('--topk-time-step', type=int, default=1,
                            help='consider the top k time step of the estimator')
        parser.add_argument('--estimator-hidden-dim', type=int, default=512,
                            help='dim of the hidden vector in the estimator model')
        parser.add_argument('--xml-model-path', type=str, default='xml_model/mlm_tlm_xnli15_1024.pth',
                            help='the pretrained xml model')
        parser.add_argument('--xml-tgt-only', type=str, default=1,
                            help='only use tgt information in xml model')
        parser.add_argument('--loss-combine', type=float, default=0,
                            help='combine the xml model and the trf model')
        # fmt: on

    def __init__(self, args, src_dict, tgt_dict):
        if args.method == 'sMoElp':
            # soft MoE with learned prior
            self.uniform_prior = False
            self.hard_selection = False
        elif args.method == 'sMoEup':
            # soft MoE with uniform prior
            self.uniform_prior = True
            self.hard_selection = False
        elif args.method == 'hMoElp':
            # hard MoE with learned prior
            self.uniform_prior = False
            self.hard_selection = True
        elif args.method == 'hMoEup':
            # hard MoE with uniform prior
            self.uniform_prior = True
            self.hard_selection = True

        if args.share_estimator > 0:
            self.share_estimator = True
        else:
            self.share_estimator = False
        if args.estimator_xml_only > 0:
            self.estimator_xml_only = True
        else:
            self.estimator_xml_only = False

        if args.share_xml_dict > 0:
            self.share_xml_dict = True
        else:
            self.share_xml_dict = False

        if int(args.xml_tgt_only) > 0:
            self.xml_tgt_only = True
        else:
            self.xml_tgt_only = False

        self.loss_combine = args.loss_combine

        self.topk_time_step = args.topk_time_step
        self.estimator_hidden_dim = args.estimator_hidden_dim

            # add indicator tokens for each expert
        for i in range(args.num_experts):
            # add to both dictionaries in case we're sharing embeddings
            src_dict.add_symbol('<expert_{}>'.format(i))
            tgt_dict.add_symbol('<expert_{}>'.format(i))

        self.loss_fn = torch.nn.MSELoss(reduce=False, size_average=False)

        super().__init__(args, src_dict, tgt_dict)

    def build_model(self, args):
        from fairseq import models
        model = models.build_model(args, self)
        xml_estimator = None
        estimator = None

        if not self.uniform_prior and not hasattr(model, 'gating_network'):
            if self.args.mean_pool_gating_network:
                if getattr(args, 'mean_pool_gating_network_encoder_dim', None):
                    encoder_dim = args.mean_pool_gating_network_encoder_dim
                elif getattr(args, 'encoder_embed_dim', None):
                    # assume that encoder_embed_dim is the encoder's output dimension
                    encoder_dim = args.encoder_embed_dim
                else:
                    raise ValueError('Must specify --mean-pool-gating-network-encoder-dim')

                if getattr(args, 'mean_pool_gating_network_dropout', None):
                    dropout = args.mean_pool_gating_network_dropout
                elif getattr(args, 'dropout', None):
                    dropout = args.dropout
                else:
                    raise ValueError('Must specify --mean-pool-gating-network-dropout')

                model.gating_network = modules.MeanPoolGatingNetwork(
                    encoder_dim, args.num_experts, dropout,
                )
            else:
                raise ValueError(
                    'translation_moe task with learned prior requires the model to '
                    'have a gating network; try using --mean-pool-gating-network'
                )
        if self.share_xml_dict:
            estimator = Estimator(self.estimator_hidden_dim, args.estimator_xml_dim + args.estimator_transformer_dim,
                                  dropout, topk_time_step=self.topk_time_step)
        elif self.estimator_xml_only:
            estimator = Estimator(self.estimator_hidden_dim, args.estimator_xml_dim, dropout,
                                  topk_time_step=self.topk_time_step)
        elif args.estimator_transformer_dim != 0:
            if self.share_estimator:
                estimator = Estimator(self.estimator_hidden_dim, args.estimator_transformer_dim, dropout,
                                      share_estimator=True, topk_time_step=self.topk_time_step)
            else:
                estimator = Estimator(self.estimator_hidden_dim, args.estimator_transformer_dim, dropout,
                                      topk_time_step=self.topk_time_step)
            if args.estimator_xml_dim != 0:
                xml_estimator = Estimator(self.estimator_hidden_dim, args.estimator_xml_dim, dropout,
                                          topk_time_step=self.topk_time_step)
        else:
            raise ValueError(
                'translation_moe task with learned prior requires the model to '
                'have a gating network; try using --mean-pool-gating-network'
            )

        return model, estimator, xml_estimator

    def quantile_loss(self, prediction, ground_truth, weight):
        marg = ground_truth - prediction
        pos = torch.gt(marg, 0).float()
        neg = torch.le(marg, 0).float()
        return ((1-weight)*pos - weight*neg)*marg

    def expert_index(self, i):
        return i + self.tgt_dict.index('<expert_0>')

    def get_ter_prediction(self, predictor, estimator, sample, xml_model=None):
        ''' get ter prediction '''
        k = self.args.num_experts

        def get_gap_fea():
            ''' get NMT features '''
            predictor.eval()
            encoder = predictor.encoder
            encoder_out = torch.zeros((sample['net_input']['src_tokens'].shape[1],
                                       sample['net_input']['src_tokens'].shape[0], encoder.output_embed_dim)).to(sample['target'].device)
            encoder_padding_mask = sample['net_input']['src_tokens'].eq(encoder.embed_positions.padding_idx)
            if not encoder_padding_mask.any():
                encoder_padding_mask = None
            encoder_out = encoder.encode(sample['net_input']['src_tokens'],
                                         encoder_out={'encoder_out': encoder_out, 'encoder_padding_mask': encoder_padding_mask})

            net_outputs = []
            i_equals = []
            for i in range(k):
                decoder = predictor.decoder
                prev_output_tokens_k = sample['net_input']['prev_output_tokens'].clone()

                assert not prev_output_tokens_k.requires_grad
                prev_output_tokens_k[:, 0] = self.expert_index(i)

                # model derived features and dual model features
                net_output = predictor.decoder(prev_output_tokens_k, encoder_out) # B x T x dic_size
                lprobs = predictor.get_normalized_probs(net_output, log_probs=True)

                target = sample['target']
                co_attn = torch.zeros((sample['target'].shape[1], sample['target'].shape[0], predictor.encoder.output_embed_dim)).to(sample['target'].device)
                encoder_padding_mask = sample['target'].eq(predictor.encoder.embed_positions.padding_idx)
                if not encoder_padding_mask.any():
                    encoder_padding_mask = None
                enc_out_dual = decoder.encode(sample['target'],
                                                        encoder_out={'encoder_out': co_attn, 'encoder_padding_mask': encoder_padding_mask})
                enc_out_dual = enc_out_dual['encoder_out'].transpose(0, 1)
                lprobs_dual = decoder.output_layer(enc_out_dual)
                lprobs_dual = utils.log_softmax(lprobs_dual, dim=-1)

                target_embeding = F.embedding(target, predictor.decoder.embed_tokens.weight)
                last_output = net_output[1]['last_output'] * target_embeding
                pre_qefv = torch.mul(last_output, target_embeding)
                post_qefv = last_output
                pre_qefv_dual = enc_out_dual * target_embeding * target_embeding
                post_qefv_dual = enc_out_dual * target_embeding

                # mismatch features
                target = target.unsqueeze(-1)
                i_gt = lprobs.gather(dim=-1, index=target)
                i_max, i_argmax = lprobs.max(dim=-1, keepdim=True)
                i_equal = torch.eq(i_argmax, target).type_as(i_gt)
                i_equals.append(i_equal)
                i_gap = i_max - i_gt

                # i_gt_dual = lprobs_dual.gather(dim=-1, index=target)
                # i_max_dual, i_argmax_dual = lprobs_dual.max(dim=-1, keepdim=True)

                mismatch_fea = torch.cat([i_gt, i_max, i_equal, i_gap,], dim=-1)
                                     # i_gt_dual, i_max_dual, i_equal_dual, i_gap_dual,
                                     # i_gt-i_gt_dual, i_max-i_max_dual, i_equal-i_equal_dual, i_gap-i_gap_dual], dim=-1)

                net_outputs.append(mismatch_fea)
                net_outputs.append(post_qefv)
                net_outputs.append(pre_qefv)
                net_outputs.append(post_qefv_dual)
                net_outputs.append(pre_qefv_dual)

            net_outputs = torch.cat(net_outputs, dim=-1)  # -> B x K

            mask = 1 - torch.eq(sample['target'], 1).unsqueeze(dim=-1).type_as(net_outputs)
            mask = mask.repeat(1, 1, net_outputs.shape[-1])
            net_outputs = net_outputs * mask
            return net_outputs

        # NMT features
        mt_qefv_prim = get_gap_fea()

        xml_model.eval()

        with torch.no_grad():
            tensor = xml_model('fwd', x=sample['xml_word_ids'], lengths=sample['xml_lengths'], langs=sample['langs'].cuda(), causal=False).contiguous()

        if self.xml_tgt_only:
            ''' only extract target features for XLM '''
            xml_src_lengths = sample['xml_src_lengths']
            xml_tgt_lengths = sample['xml_tgt_lengths']

            tensor_ = tensor.transpose(0, 1)
            tensor = torch.unbind(tensor_)
            xml_word_ids = sample['xml_word_ids'].transpose(0, 1)
            xml_word_ids = torch.unbind(xml_word_ids)

            max_tgt_length = max(xml_tgt_lengths)
            max_tgt_length = max(max_tgt_length, mt_qefv_prim.shape[1])
            xml_tensor = torch.FloatTensor(len(tensor), max_tgt_length, 1024).fill_(0)
            xml_tgt_word_ids = torch.LongTensor(len(tensor), max_tgt_length).fill_(2)
            for i, (t, tgt_word_id)  in enumerate(zip(tensor, xml_word_ids)):
                start = xml_src_lengths[i] + 3
                end = start + xml_tgt_lengths[i]
                selected_tensor = t[start : end]
                xml_tensor[i, :selected_tensor.shape[0]] = selected_tensor
                selected_tgt_word_ids = tgt_word_id[start : end]
                xml_tgt_word_ids[i, :selected_tensor.shape[0]] = selected_tgt_word_ids

            mask = torch.ne(xml_tgt_word_ids, 2).cuda().float()
            target_embeding = F.embedding(xml_tgt_word_ids.cuda(), xml_model.pred_layer.proj.weight)

            xml_tensor = xml_tensor.cuda() * (mask.unsqueeze(-1).expand_as(xml_tensor)) * target_embeding
            pre_qefv = torch.mul(xml_tensor, target_embeding)
            post_qefv = xml_tensor
            pre_qefv = torch.tanh(estimator.reduce_dim(pre_qefv))
            post_qefv = torch.tanh(estimator.reduce_dim(post_qefv))
            paded_tensor = torch.FloatTensor(xml_tensor.shape[0], xml_tensor.shape[1], 4).fill_(0).cuda()

            prob = xml_model.pred_layer.proj(xml_tensor.cuda())
            prob = utils.log_softmax(prob, dim=-1)
            target = xml_tgt_word_ids.unsqueeze(-1).to(mt_qefv_prim.device)
            i_gt = prob.gather(dim=-1, index=target)
            i_max, i_argmax = prob.max(dim=-1, keepdim=True)
            i_equal = torch.eq(i_argmax, target).type_as(i_gt)
            i_gap = i_max - i_gt
            gap_fea = torch.cat([i_gt, i_max, i_equal, i_gap,], dim=-1)

            xml_qefv = torch.cat([gap_fea, post_qefv, pre_qefv, post_qefv, pre_qefv], dim=-1)
        else:
            ''' extract both source and target features for XLM '''
            xml_tensor = tensor.transpose(0, 1)
            xml_word_ids = sample['xml_word_ids'].transpose(0, 1)

            mask = torch.ne(xml_word_ids, 2).cuda().float()
            target_embeding = F.embedding(xml_word_ids.cuda(), xml_model.pred_layer.proj.weight)
            xml_tensor = xml_tensor.cuda() * (mask.unsqueeze(-1).expand_as(xml_tensor)) * target_embeding
            pre_qefv = torch.mul(xml_tensor, target_embeding)
            post_qefv = xml_tensor
            pre_qefv = torch.tanh(estimator.reduce_dim(pre_qefv))
            post_qefv = torch.tanh(estimator.reduce_dim(post_qefv))
            paded_tensor = torch.FloatTensor(xml_tensor.shape[0], xml_tensor.shape[1], 4).fill_(0).cuda()

            prob = xml_model.pred_layer.proj(tensor.transpose(0, 1))
            prob = utils.log_softmax(prob, dim=-1)
            target = xml_word_ids.unsqueeze(-1).to(mt_qefv_prim.device)
            i_gt = prob.gather(dim=-1, index=target)
            i_max, i_argmax = prob.max(dim=-1, keepdim=True)
            i_equal = torch.eq(i_argmax, target).type_as(i_gt)
            i_gap = i_max - i_gt
            gap_fea = torch.cat([i_gt, i_max, i_equal, i_gap,], dim=-1)

            xml_qefv = torch.cat([gap_fea, post_qefv, pre_qefv, post_qefv, pre_qefv], dim=-1)  # .transpose(0, 1)


        ter_prediction = 0

        if self.share_xml_dict:
            ter_prediction = estimator(torch.cat([xml_qefv, mt_qefv_prim], dim=-1))
        else:
            if self.estimator_xml_only:
                xml_qefv = torch.cat([pre_qefv, pre_qefv, pre_qefv, pre_qefv, pre_qefv, paded_tensor], dim=-1)
                ter_prediction += estimator(xml_qefv)
            else:
                xml_qefv = torch.cat([xml_qefv, xml_qefv, xml_qefv, xml_qefv, xml_qefv], dim=-1)
                ter_prediction += estimator.combine_forward(mt_qefv_prim, xml_qefv)

        return ter_prediction

    def _get_qe_loss(self, sample, predictor, estimator, criterion, xml_model=None, valid=False):
        assert hasattr(criterion, 'compute_loss'), \
            'translation_moe task requires the criterion to implement the compute_loss() method'

        k = self.args.num_experts
        bsz = sample['target'].size(0)

        # compute loss with dropout
        ter_prediction = self.get_ter_prediction(predictor, estimator, sample, xml_model=xml_model).squeeze(-1)
        ter_gt = sample['ter']
        loss = self.loss_fn(ter_prediction, ter_gt)
        loss = loss.sum()
        sample_size = bsz
        logging_output = {
            'loss': utils.item(loss if valid else loss.data),
            'ntokens': sample['ntokens'],
            'sample_size': sample_size,
            'posterior':ter_prediction.cpu(),
        }
        return loss, sample_size, logging_output

    def _get_loss(self, sample, model, criterion):
        assert hasattr(criterion, 'compute_loss'), \
            'translation_moe task requires the criterion to implement the compute_loss() method'

        k = self.args.num_experts
        bsz = sample['target'].size(0)

        def get_lprob_y(encoder_out, prev_output_tokens_k):
            net_output = model.decoder(prev_output_tokens_k, encoder_out)
            loss, _ = criterion.compute_loss(model, net_output, sample, reduce=False)
            loss = loss.view(bsz, -1)
            return -loss.sum(dim=1, keepdim=True)  # -> B x 1

        def get_lprob_yz(winners=None):
            ''' mixture models for NMT '''
            encoder_out = model.encoder(sample['net_input']['src_tokens'], sample['net_input']['src_lengths'])

            if winners is None:
                lprob_y = []
                for i in range(k):
                    prev_output_tokens_k = sample['net_input']['prev_output_tokens'].clone()
                    assert not prev_output_tokens_k.requires_grad
                    prev_output_tokens_k[:, 0] = self.expert_index(i)
                    lprob_y.append(get_lprob_y(encoder_out, prev_output_tokens_k))
                lprob_y = torch.cat(lprob_y, dim=1)  # -> B x K
            else:
                prev_output_tokens_k = sample['net_input']['prev_output_tokens'].clone()
                prev_output_tokens_k[:, 0] = self.expert_index(winners)
                lprob_y = get_lprob_y(encoder_out, prev_output_tokens_k)  # -> B

            if self.uniform_prior:
                lprob_yz = lprob_y
            else:
                lprob_z = model.gating_network(encoder_out)  # B x K
                if winners is not None:
                    lprob_z = lprob_z.gather(dim=1, index=winners.unsqueeze(-1))
                lprob_yz = lprob_y + lprob_z.type_as(lprob_y)  # B x K

            return lprob_yz

        # compute responsibilities without dropout
        with eval(model):  # disable dropout
            with torch.no_grad():  # disable autograd
                lprob_yz = get_lprob_yz()  # B x K
                prob_z_xy = torch.nn.functional.softmax(lprob_yz, dim=1)
        assert not prob_z_xy.requires_grad

        # compute loss with dropout
        if self.hard_selection:
            winners = prob_z_xy.max(dim=1)[1]
            loss = -get_lprob_yz(winners)
        else:
            lprob_yz = get_lprob_yz()  # B x K
            loss = -modules.LogSumExpMoE.apply(lprob_yz, prob_z_xy, 1)

        loss = loss.sum()
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data),
            'ntokens': sample['ntokens'],
            'sample_size': sample_size,
            'posterior': prob_z_xy.float().sum(dim=0).cpu(),
        }
        return loss, sample_size, logging_output

    def train_step(self, sample, predictor, estimator, criterion, optimizer, ignore_grad=False, xml_model=None):
        estimator.train()
        loss, sample_size, logging_output = self._get_qe_loss(sample, predictor, estimator, criterion,
                                                              xml_model=xml_model)
        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)
        return loss, sample_size, logging_output

    def valid_step(self, sample, predictor, estimator, criterion, xml_model=None):
        predictor.eval()
        estimator.eval()
        with torch.no_grad():
            loss, sample_size, logging_output = self._get_qe_loss(sample, predictor, estimator, criterion,
                                                                  xml_model=xml_model, valid=True)
        return loss, sample_size, logging_output

    def inference_step(self, generator, models, sample, prefix_tokens=None, expert=None):
        expert = expert or self.args.gen_expert
        with torch.no_grad():
            return generator.generate(
                models,
                sample,
                prefix_tokens=prefix_tokens,
                bos_token=self.expert_index(expert),
            )

    def aggregate_logging_outputs(self, logging_outputs, criterion):
        agg_logging_outputs = criterion.__class__.aggregate_logging_outputs(logging_outputs)
        agg_logging_outputs['posterior'] = sum(
            log['posterior'] for log in logging_outputs if 'posterior' in log
        )
        return agg_logging_outputs
