# Beam Generator class
import torch
from addict import Dict

from codes.utils.data import START_TOKEN, END_TOKEN
from codes.utils.util import tile
from codes.net.onmt.beam import *

class Generator:
    def __init__(self, data, model, config, trainer=None):
        """
        :param data: reference to DataUtility
        :param model:
            :param encoder:
            :param decoder:
        :config
        """
        self.data = data
        self.encoder_model = model.encoder
        self.decoder_model = model.decoder
        self.beam_size = config.model.beam.beam_size
        self.config = config
        # pointer to Trainer class
        self.trainer = trainer

        # beam scorer
        self.scorer = GNMTGlobalScorer(config.model.beam.alpha,
                                       config.model.beam.beta,
                                       config.model.beam.coverage_penalty,
                                       config.model.beam.length_penalty)

    def beam_process_batch(self, batch, max_length, min_length=0, n_best=1,
                           return_attention=False):
        """
        Wrapper around Beam Search
        heavily borrowed from https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/translate/translator.py#L316
        :return:
        """

        beam_size = self.beam_size
        batch_size = batch.batch_size
        vocab = self.data.word2id
        start_token = vocab[START_TOKEN]
        end_token = vocab[END_TOKEN]

        # Encoder forward
        inp, inp_lengths = batch.inp, batch.inp_lengths
        inp_lengths = torch.LongTensor(inp_lengths).to(batch.inp.device)
        encoder_outputs, encoder_hidden = self.encoder_model(inp, inp_lengths)
        decoder_states = self.decoder_model.calculate_hidden(
            batch.batch_size, encoder_outputs, encoder_hidden, batch.outp_ents)

        # Tile states and memory beam_size times
        # multiply the batches
        dec_h, dec_c = decoder_states
        dec_h = tile(dec_h, beam_size, dim=1)
        dec_c = tile(dec_c, beam_size, dim=1)
        decoder_states = (dec_h, dec_c)
        encoder_outputs = tile(encoder_outputs, beam_size, dim=0)
        inp_lengths = tile(inp_lengths, beam_size, dim=0)

        batch_offset = torch.arange(
            batch_size, dtype=torch.long, device=encoder_outputs.device)
        beam_offset = torch.arange(
            0,
            batch_size * beam_size,
            step = beam_size,
            dtype = torch.long,
            device = encoder_outputs.device)

        alive_seq = torch.full(
            [batch_size * beam_size, 1],
            start_token,
            dtype=torch.long,
            device=encoder_outputs.device)

        alive_attn = None

        # Give full probability to the first beam on the first step.
        topk_log_probs = (
            torch.tensor([0.0] + [float("-inf")] * (beam_size - 1),
                         device=encoder_outputs.device).repeat(batch_size))

        results = Dict()
        results.predictions = [[] for _ in range(batch_size)]  # noqa: F812
        results.scores = [[] for _ in range(batch_size)]  # noqa: F812
        results.attention = [[] for _ in range(batch_size)]  # noqa: F812
        results.gold_score = [0] * batch_size
        results.batch = batch

        max_length += 1

        for step in range(max_length):
            decoder_input = alive_seq[:, -1].view(-1, 1) # Batch x Seq

            # Decoder forward
            log_probs, attn, dec_states = self.decoder_model(
                decoder_input,
                decoder_states,
                encoder_outputs=encoder_outputs,
                encoder_lengths=inp_lengths)
            vocab_size = log_probs.size(-1)
            log_probs = log_probs.squeeze(1) # B x vocab

            if step < min_length:
                log_probs[:, end_token] = -1e20

            # Multiply probs by the beam probability.
            log_probs += topk_log_probs.view(-1).unsqueeze(1)

            alpha = self.scorer.alpha
            length_penalty = ((5.0 + (step + 1)) / 6.0) ** alpha

            # Flatten probs into a list of possibilities.
            curr_scores = log_probs / length_penalty
            curr_scores = curr_scores.reshape(-1, beam_size * vocab_size)
            topk_scores, topk_ids = curr_scores.topk(beam_size, dim=-1)

            # Recover log probs.
            topk_log_probs = topk_scores * length_penalty

            # Resolve beam origin and true word ids.
            topk_beam_index = topk_ids.div(vocab_size)
            topk_ids = topk_ids.fmod(vocab_size)

            # Map beam_index to batch_index in the flat representation.
            batch_index = (
                    topk_beam_index
                    + beam_offset[:topk_beam_index.size(0)].unsqueeze(1))

            # End condition is the top beam reached end_token.
            end_condition = topk_ids[:, 0].eq(end_token)
            if step + 1 == max_length:
                end_condition.fill_(1)
            finished = end_condition.nonzero().view(-1)

            # Save result of finished sentences.
            if len(finished) > 0:
                predictions = alive_seq.view(beam_size, -1,  alive_seq.size(-1))
                scores = topk_scores.view(-1, beam_size)
                attention = None
                if alive_attn is not None:
                    attention = alive_attn.view(
                        alive_attn.size(0), -1, beam_size, alive_attn.size(-1))
                for i in finished:
                    b = batch_offset[i]
                    for n in range(n_best):
                        results.predictions[b].append(predictions[n, i, 1:])
                        results.scores[b].append(scores[i, n])
                        if attention is None:
                            results.attention[b].append([])
                        else:
                            results.attention[b].append(
                                attention[:, i, n, :inp_lengths[i]])
                non_finished = end_condition.eq(0).nonzero().view(-1)
                # If all sentences are translated, no need to go further.
                if len(non_finished) == 0:
                    break
                # Remove finished batches for the next step.
                topk_log_probs = topk_log_probs.index_select(
                    0, non_finished.to(topk_log_probs.device))
                topk_ids = topk_ids.index_select(0, non_finished)
                batch_index = batch_index.index_select(0, non_finished)
                batch_offset = batch_offset.index_select(0, non_finished)

            # Select and reorder alive batches.
            select_indices = batch_index.view(-1)
            alive_seq = alive_seq.index_select(0, select_indices)
            encoder_outputs = encoder_outputs.index_select(0, select_indices)
            inp_lengths = inp_lengths.index_select(0, select_indices)
            decoder_states = map_batch_fn(decoder_states,
                lambda state, dim: state.index_select(dim, select_indices))

            # Append last prediction.
            alive_seq = torch.cat([alive_seq, topk_ids.view(-1, 1)], -1)

            if return_attention:
                current_attn = attn["std"].index_select(1, select_indices)
                if alive_attn is None:
                    alive_attn = current_attn
                else:
                    alive_attn = alive_attn.index_select(1, select_indices)
                    alive_attn = torch.cat([alive_attn, current_attn], 0)

        return results

    def process_batch(self, batch, predictions=None, mode='train', beam=False):
        """
        Given a batch of input/output pairs, generate the true text for input,
        true output and generated output
        :param beam: if True, use beam search, else use the trainer
        :return:
        """
        if beam:
            beam_results = self.beam_process_batch(batch,
                        max_length=self.config.model.beam.max_length,
                        n_best=self.config.model.beam.n_best)
            predictions = beam_results.predictions
            # select the top prediction only for now
            predictions = [row[0] for row in predictions]
        else:
            assert predictions is not None
        predictions = predictions.tolist()
        if batch.inp.dim() == 3:
            # convert sentences to whole passage
            inp = batch.inp.view(batch.batch_size, -1)
        else:
            inp = batch.inp
        inp = inp.view(batch.batch_size, -1)
        batch.true_inp = self._convert_mat_to_text(inp)
        batch.true_outp = self._convert_mat_to_text(batch.target, target=True)
        batch.pred_outp = self._convert_mat_to_text(predictions, target=True)
        return batch

    def _convert_mat_to_text(self, tensor, target=False):
        # TODO: handle 3 dimensional case
        if type(tensor) != list:
            assert tensor.dim() == 2
        else:
            if type(tensor[0]) != list:
                return self._convert_list_to_text(tensor, target=target)

        id2word = self.data.id2word
        if target:
            id2word = self.data.target_id2word
        return [[id2word[int(w)] if int(w) in id2word else "<s>" for w in sent] for sent in tensor]

    def _convert_list_to_text(self, tlist, target=False):
        id2word = self.data.id2word
        if target:
            id2word = self.data.target_id2word
        return [[id2word[int(w)]] if int(w) in id2word else "<s>" for w in tlist]




def map_batch_fn(hidden, fn):
    hidden = tuple(map(lambda x: fn(x, 1), hidden))
    return hidden