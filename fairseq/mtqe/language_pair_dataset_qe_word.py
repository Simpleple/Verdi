# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import numpy as np
import torch

from ..data import data_utils, FairseqDataset


def collate(
    samples, pad_idx, eos_idx, xml_pad_indx=None, left_pad_source=True, left_pad_target=False,
    input_feeding=True,
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    id = torch.LongTensor([s['id'] for s in samples])
    src_tokens = merge('source', left_pad=left_pad_source)
    # sort by descending source length
    src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
    tgt_lengths = torch.LongTensor([s['target'].numel() for s in samples])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    prev_output_tokens = None
    target = None
    if samples[0].get('target', None) is not None:
        target = merge('target', left_pad=left_pad_target)
        target = target.index_select(0, sort_order)
        ntokens = sum(len(s['target']) for s in samples)

        if input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                'target',
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
    else:
        ntokens = sum(len(s['source']) for s in samples)

    ter = torch.Tensor([s['ter'] for s in samples]).index_select(0, sort_order)
    tgt_lengths = tgt_lengths.index_select(0, sort_order)

    ## xml data
    xml_src_lengths = []
    xml_tgt_lengths = []
    xml_tgt_word_id_list = []
    xml_lengths = [sent['xml'].shape[0] for sent in samples]
    slen = max(xml_lengths)
    word_ids = torch.LongTensor(slen, len(samples)).fill_(xml_pad_indx)
    pred_mask = torch.LongTensor(slen, len(samples)).fill_(0)
    for i in range(len(samples)):
        sentence = samples[i]['xml']
        word_ids[:sentence.shape[0], i] = sentence
        mask_sentences = torch.ByteTensor(samples[i]['xml_mask_sentence'])
        pred_mask[:mask_sentences.shape[0], i] = mask_sentences

        xml_ori = samples[i]['xml_ori'].split('</s> </s>')
        xml_src_length = len(xml_ori[0].strip().split()) - 1
        xml_src_lengths.append(xml_src_length)
        xml_tgt_length = len(xml_ori[1].strip().split()) - 1
        xml_tgt_lengths.append(xml_tgt_length)
        start  = xml_src_length + 2
        end = start + xml_tgt_length
        xml_tgt_word_id = sentence[start:end]
        xml_tgt_word_id_list.append(xml_tgt_word_id)

    max_tgt_leng = max(xml_tgt_lengths)
    slen_ = max([x+3+max_tgt_leng for x in xml_src_lengths ])
    slen = max(slen, slen_)
    xml_tgt_word_ids = torch.LongTensor(max_tgt_leng, len(samples)).fill_(xml_pad_indx)
    pred_mask_new = torch.ByteTensor(slen, len(samples)).fill_(0)
    word_ids = torch.LongTensor(slen, len(samples)).fill_(xml_pad_indx)
    langs = torch.LongTensor(slen, len(samples)).fill_(4)  # en langid
    for i in range(len(xml_tgt_word_id_list)):
        tgt = xml_tgt_word_id_list[i]
        xml_tgt_word_ids[:tgt.shape[0], i] = tgt

        src_mask = [0] * (xml_src_lengths[i] + 3)
        tgt_mask = [1] * (max_tgt_leng)
        src_mask.extend(tgt_mask)
        pred_mask_new[:len(src_mask), i] = torch.ByteTensor(src_mask)

        langs[:(xml_src_lengths[i] + 2), i] = torch.LongTensor([14] * (xml_src_lengths[i] + 2))

        sentence = samples[i]['xml']
        word_ids[:sentence.shape[0], i] = sentence

    max_bpe_len = max([len(s['bpe_tag']) for s in samples])
    batch_word_tags = torch.zeros([len(samples), max_bpe_len]).fill_(-1) # no eos token
    batch_gap_tags = torch.zeros([len(samples), max_bpe_len + 1]).fill_(-1)
    batch_bpe_tags = []
    batch_xml_bpe_tags = []
    max_src_bpe_len = max([len(s['src_bpe_tag']) for s in samples])
    batch_src_word_tags = torch.zeros([len(samples), max_src_bpe_len]).fill_(-1)
    batch_src_bpe_tags = []
    batch_xml_src_bpe_tags = []
    for i, order in enumerate(sort_order):
        order = order.item()
        batch_word_tags[i, :len(samples[order]['word_tag'])] = torch.Tensor(samples[order]['word_tag'])
        batch_gap_tags[i, :len(samples[order]['gap_tag'])] = torch.Tensor(samples[order]['gap_tag'])
        batch_bpe_tags.append(samples[order]['bpe_tag'])
        batch_xml_bpe_tags.append(samples[order]['xml_bpe_tag'])
        batch_src_word_tags[i, :len(samples[order]['src_word_tag'])] = torch.Tensor(samples[order]['src_word_tag'])
        batch_src_bpe_tags.append(samples[order]['src_bpe_tag'])
        batch_xml_src_bpe_tags.append(samples[order]['xml_src_bpe_tag'])

    word_ids = word_ids.index_select(1, sort_order)
    pred_mask_new = pred_mask_new.index_select(1, sort_order)
    xml_lengths = torch.LongTensor(xml_lengths).index_select(0, sort_order)
    xml_src_lengths = torch.LongTensor(xml_src_lengths).index_select(0, sort_order)
    xml_tgt_lengths = torch.LongTensor(xml_tgt_lengths).index_select(0, sort_order)
    xml_tgt_word_ids = xml_tgt_word_ids.index_select(1, sort_order)
    langs = langs.index_select(1, sort_order)

    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
            'tgt_lengths': tgt_lengths,
        },
        'target': target,
        'ter': ter,
        'xml_word_ids': word_ids,
        # 'xml_pred_mask': pred_mask_new,
        'xml_lengths': torch.LongTensor(xml_lengths),
        # 'xml_ori': samples['xml_ori'],
        'xml_src_lengths': torch.LongTensor(xml_src_lengths),
        'xml_tgt_lengths': torch.LongTensor(xml_tgt_lengths),
        'xml_tgt_word_ids': xml_tgt_word_ids,
        'langs': langs,
        'word_tags': batch_word_tags,
        'gap_tags': batch_gap_tags,
        'bpe_tags': batch_bpe_tags,
        'xml_bpe_tags': batch_xml_bpe_tags,
        'src_word_tags': batch_src_word_tags,
        'src_bpe_tags': batch_src_bpe_tags,
        'xml_src_bpe_tags': batch_xml_src_bpe_tags,
    }
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens
    return batch

def collate_no_shuffle(
    samples, pad_idx, eos_idx, xml_pad_indx=None, left_pad_source=True, left_pad_target=False,
    input_feeding=True,
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    id = torch.LongTensor([s['id'] for s in samples])
    src_tokens = merge('source', left_pad=left_pad_source)
    # sort by descending source length
    src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
    tgt_lengths = torch.LongTensor([s['target'].numel() for s in samples])
    # src_lengths, sort_order = src_lengths.sort(descending=True)
    # id = id.index_select(0, sort_order)
    # src_tokens = src_tokens.index_select(0, sort_order)

    prev_output_tokens = None
    target = None
    if samples[0].get('target', None) is not None:
        target = merge('target', left_pad=left_pad_target)
        # target = target.index_select(0, sort_order)
        ntokens = sum(len(s['target']) for s in samples)

        if input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                'target',
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
            )
            # prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
    else:
        ntokens = sum(len(s['source']) for s in samples)

    # ter = torch.Tensor([s['ter'] for s in samples]).index_select(0, sort_order)
    # tgt_lengths = tgt_lengths.index_select(0, sort_order)
    ter = torch.Tensor([s['ter'] for s in samples])

    ## xml data
    xml_lengths = [sent['xml'].shape[0] for sent in samples]
    slen = max(xml_lengths)
    word_ids = torch.LongTensor(slen, len(samples)).fill_(xml_pad_indx)
    pred_mask = torch.ByteTensor(slen, len(samples)).fill_(0)
    xml_src_lengths = []
    xml_tgt_lengths = []
    xml_tgt_word_id_list = []

    for i in range(len(samples)):
        sentence = samples[i]['xml']
        word_ids[:sentence.shape[0], i] = sentence
        mask_sentences = torch.LongTensor(samples[i]['xml_mask_sentence'])
        pred_mask[:mask_sentences.shape[0], i] = mask_sentences
        xml_ori = samples[i]['xml_ori'].split('</s> </s>')
        xml_src_length = len(xml_ori[0].strip().split())
        xml_src_lengths.append(xml_src_length)
        xml_tgt_length = len(xml_ori[1].strip().split())
        xml_tgt_lengths.append(xml_tgt_length)
        start = xml_src_length + 2
        end = start + xml_tgt_length
        xml_tgt_word_id = sentence[start:end]
        xml_tgt_word_id_list.append(xml_tgt_word_id)

    max_tgt_leng = max(xml_tgt_lengths)
    slen_ = max([x + 3 + max_tgt_leng for x in xml_src_lengths])
    slen = max(slen, slen_)
    xml_tgt_word_ids = torch.LongTensor(max_tgt_leng, len(samples)).fill_(xml_pad_indx)
    pred_mask_new = torch.ByteTensor(slen, len(samples)).fill_(0)
    word_ids = torch.LongTensor(slen, len(samples)).fill_(xml_pad_indx)
    langs = torch.LongTensor(slen, len(samples)).fill_(4)  # en langid
    for i in range(len(xml_tgt_word_id_list)):
        tgt = xml_tgt_word_id_list[i]
        xml_tgt_word_ids[:tgt.shape[0], i] = tgt

        src_mask = [0] * (xml_src_lengths[i] + 3)
        tgt_mask = [1] * (max_tgt_leng)
        src_mask.extend(tgt_mask)
        pred_mask_new[:len(src_mask), i] = torch.ByteTensor(src_mask)

        langs[:(xml_src_lengths[i] + 2), i] = torch.LongTensor([14] * (xml_src_lengths[i] + 2))

        sentence = samples[i]['xml']
        word_ids[:sentence.shape[0], i] = sentence

    max_bpe_len = max([len(s['bpe_tag']) for s in samples])
    batch_word_tags = torch.zeros([len(samples), max_bpe_len]).fill_(-1) # no eos token
    batch_gap_tags = torch.zeros([len(samples), max_bpe_len + 1]).fill_(-1)
    batch_bpe_tags = []
    batch_xml_bpe_tags = []
    max_src_bpe_len = max([len(s['src_bpe_tag']) for s in samples])
    batch_src_word_tags = torch.zeros([len(samples), max_src_bpe_len]).fill_(-1)
    batch_src_bpe_tags = []
    batch_xml_src_bpe_tags = []
    for i in range(len(samples)):
        batch_word_tags[i, :len(samples[i]['word_tag'])] = torch.Tensor(samples[i]['word_tag'])
        batch_gap_tags[i, :len(samples[i]['gap_tag'])] = torch.Tensor(samples[i]['gap_tag'])
        batch_bpe_tags.append(samples[i]['bpe_tag'])
        batch_xml_bpe_tags.append(samples[i]['xml_bpe_tag'])
        batch_src_word_tags[i, :len(samples[i]['src_word_tag'])] = torch.Tensor(samples[i]['src_word_tag'])
        batch_src_bpe_tags.append(samples[i]['src_bpe_tag'])
        batch_xml_src_bpe_tags.append(samples[i]['xml_src_bpe_tag'])

    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
            'tgt_lengths': tgt_lengths,
        },
        'target': target,
        'ter': ter,
        'xml_word_ids': word_ids,
        # 'xml_pred_mask': pred_mask_new,
        'xml_lengths': torch.LongTensor(xml_lengths),
        # 'xml_ori': samples['xml_ori'],
        'xml_src_lengths': torch.LongTensor(xml_src_lengths),
        'xml_tgt_lengths': torch.LongTensor(xml_tgt_lengths),
        'xml_tgt_word_ids': xml_tgt_word_ids,
        'langs': langs,
        'word_tags': batch_word_tags,
        'gap_tags': batch_gap_tags,
        'bpe_tags': batch_bpe_tags,
        'xml_bpe_tags': batch_xml_bpe_tags,
        'src_word_tags': batch_src_word_tags,
        'src_bpe_tags': batch_src_bpe_tags,
        'xml_src_bpe_tags': batch_xml_src_bpe_tags,
    }
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens
    return batch


class LanguagePairWordDataset(FairseqDataset):
    """
    A pair of torch.utils.data.Datasets.

    Args:
        src (torch.utils.data.Dataset): source dataset to wrap
        src_sizes (List[int]): source sentence lengths
        src_dict (~fairseq.data.Dictionary): source vocabulary
        tgt (torch.utils.data.Dataset, optional): target dataset to wrap
        tgt_sizes (List[int], optional): target sentence lengths
        tgt_dict (~fairseq.data.Dictionary, optional): target vocabulary
        left_pad_source (bool, optional): pad source tensors on the left side
            (default: True).
        left_pad_target (bool, optional): pad target tensors on the left side
            (default: False).
        max_source_positions (int, optional): max number of tokens in the
            source sentence (default: 1024).
        max_target_positions (int, optional): max number of tokens in the
            target sentence (default: 1024).
        shuffle (bool, optional): shuffle dataset elements before batching
            (default: True).
        input_feeding (bool, optional): create a shifted version of the targets
            to be passed into the model for teacher forcing (default: True).
        remove_eos_from_source (bool, optional): if set, removes eos from end
            of source if it's present (default: False).
        append_eos_to_target (bool, optional): if set, appends eos to end of
            target if it's absent (default: False).
    """

    def __init__(
        self, src, src_sizes, src_dict,
        tgt=None, tgt_sizes=None, tgt_dict=None,
        ter=None, xml=None, xml_dico=None, xml_params=None, xml_pad_indx=None,
        word_tag=None, gap_tag=None, bpe_tag=None, xml_bpe_tag=None,
        src_word_tag=None, src_bpe_tag=None, xml_src_bpe_tag=None,
        left_pad_source=True, left_pad_target=False,
        max_source_positions=1024, max_target_positions=1024,
        shuffle=True, input_feeding=True, remove_eos_from_source=False, append_eos_to_target=False,
    ):
        if tgt_dict is not None:
            assert src_dict.pad() == tgt_dict.pad()
            assert src_dict.eos() == tgt_dict.eos()
            assert src_dict.unk() == tgt_dict.unk()
        self.src = src
        self.tgt = tgt
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.xml = xml
        self.xml_dico = xml_dico
        self.xml_params = xml_params
        self.xml_pad_index = xml_pad_indx
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.remove_eos_from_source = remove_eos_from_source
        self.append_eos_to_target = append_eos_to_target
        self.ter = ter
        self.word_tag = word_tag
        self.gap_tag = gap_tag
        self.bpe_tag = bpe_tag
        self.xml_bpe_tag = xml_bpe_tag
        self.src_word_tag = src_word_tag
        self.src_bpe_tag = src_bpe_tag
        self.xml_src_bpe_tag = xml_src_bpe_tag

    def __getitem__(self, index):
        tgt_item = self.tgt[index] if self.tgt is not None else None
        src_item = self.src[index]
        ter_item = self.ter[index]
        xml_item = self.xml[index]
        word_tag_item = self.word_tag[index]
        gap_tag_item = self.gap_tag[index]
        bpe_tag_item = self.bpe_tag[index]
        xml_bpe_tag_item = self.xml_bpe_tag[index]
        src_word_tag_item = self.src_word_tag[index]
        src_bpe_tag_item = self.src_bpe_tag[index]
        xml_src_bpe_tag_item = self.xml_src_bpe_tag[index]
        mask_sentence_itme = self.xml.get_masksentence(index)
        xml_ori_item = self.xml.get_original_text(index)
        # Append EOS to end of tgt sentence if it does not have an EOS and remove
        # EOS from end of src sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use tgt_dataset as src_dataset and vice versa
        if self.append_eos_to_target:
            eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
            if self.tgt and self.tgt[index][-1] != eos:
                tgt_item = torch.cat([self.tgt[index], torch.LongTensor([eos])])

        if self.remove_eos_from_source:
            eos = self.src_dict.eos()
            if self.src[index][-1] == eos:
                src_item = self.src[index][:-1]

        return {
            'id': index,
            'source': src_item,
            'target': tgt_item,
            'ter': ter_item,
            'xml': xml_item,
            'xml_mask_sentence': mask_sentence_itme,
            'xml_ori': xml_ori_item,
            'word_tag': word_tag_item,
            'gap_tag': gap_tag_item,
            'bpe_tag': bpe_tag_item,
            'xml_bpe_tag': xml_bpe_tag_item,
            'src_word_tag': src_word_tag_item,
            'src_bpe_tag': src_bpe_tag_item,
            'xml_src_bpe_tag': xml_src_bpe_tag_item,
        }

    def __len__(self):
        return len(self.src)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one
                    position for teacher forcing, of shape `(bsz, tgt_len)`.
                    This key will not be present if *input_feeding* is
                    ``False``.  Padding will appear on the left if
                    *left_pad_target* is ``True``.

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
        """
        if self.shuffle:
            return collate(
                samples, pad_idx=self.src_dict.pad(), xml_pad_indx=self.xml_pad_index, eos_idx=self.src_dict.eos(),
                left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
                input_feeding=self.input_feeding,
            )
        else:
            return collate_no_shuffle(
                samples, pad_idx=self.src_dict.pad(), xml_pad_indx=self.xml_pad_index, eos_idx=self.src_dict.eos(),
                left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
                input_feeding=self.input_feeding,
            )
    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        if self.tgt_sizes is not None:
            indices = indices[np.argsort(self.tgt_sizes[indices], kind='mergesort')]
        return indices[np.argsort(self.src_sizes[indices], kind='mergesort')]

    @property
    def supports_prefetch(self):
        return (
            getattr(self.src, 'supports_prefetch', False)
            and (getattr(self.tgt, 'supports_prefetch', False) or self.tgt is None)
            and (getattr(self.ter, 'supports_prefetch', False) or self.ter is None)
        )

    def prefetch(self, indices):
        self.src.prefetch(indices)
        if self.tgt is not None:
            self.tgt.prefetch(indices)
        if self.ter is not None:
            self.ter.prefetch(indices)
