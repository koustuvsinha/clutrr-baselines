# Sequence to Sequence net for abstractive summarization
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from codes.net.base_net import Net
from codes.utils.util import check_id_emb
import pdb

class SimpleEncoder(Net):
    """
    Simple LSTM Encoder
    """
    def __init__(self, model_config, shared_embeddings=None):
        super().__init__(model_config)

        if not shared_embeddings:
            self.init_embeddings()
        else:
            self.embedding = shared_embeddings

        self.lstm = nn.LSTM(
            model_config.embedding.dim,
            model_config.encoder.hidden_dim,
            model_config.encoder.nlayers,
            bidirectional=model_config.encoder.bidirectional,
            batch_first=True,
            dropout=model_config.encoder.dropout
        )

    def forward(self, batch):
        data = batch.inp
        data_lengths = batch.inp_lengths
        data = self.embedding(data)
        data_pack = pack_padded_sequence(data, data_lengths, batch_first=True)
        outp, hidden_rep = self.lstm(data_pack)
        outp, _ = pad_packed_sequence(outp, batch_first=True)
        outp = outp.contiguous()
        return outp.contiguous(), hidden_rep

class SimpleDecoder(Net):
    """
    Simple LSTM Decoder
    """
    def __init__(self, model_config, shared_embeddings=None, attn_module=None):
        super().__init__(model_config)

        if not shared_embeddings:
            self.init_embeddings()
        else:
            self.embedding = shared_embeddings

        if model_config.only_relation:
            base_enc_dim = model_config.embedding.dim
            if model_config.encoder.bidirectional:
                base_enc_dim *=2
            query_rep = base_enc_dim * model_config.decoder.query_ents


            input_dim = query_rep + base_enc_dim
            output_dim = model_config.vocab_size
            self.decoder2vocab = self.get_mlp(input_dim, output_dim)
        else:
            inp_dim = model_config.embedding.dim
            if model_config.encoder.bidirectional:
                inp_dim *= 2

            inp_dim *= model_config.decoder.query_ents
            inp_dim += model_config.embedding.dim

            self.lstm = nn.LSTM(
                inp_dim,
                model_config.decoder.hidden_dim,
                model_config.decoder.nlayers,
                bidirectional=model_config.decoder.bidirectional,
                batch_first=True,
                dropout=model_config.decoder.dropout
            )

            self.decoder2vocab = self.get_mlp(
                model_config.decoder.hidden_dim * self.num_directions,
                model_config.vocab_size
            )

        self.attn_module = attn_module

    def init_hidden(self, encoder_states, batch_size):
        # initial hidden state of the decoder will be an average of encoder states
        avg_state = torch.mean(encoder_states, 1, keepdim=True)  # B x 1 x dim
        avg_state = avg_state.transpose(0, 1).expand(self.model_config.decoder.nlayers, -1, -1)  # nlayers x B x dim
        encoder_hidden = []
        encoder_hidden.append(avg_state)
        encoder_hidden.append(torch.zeros_like(avg_state).to(avg_state.device))
        return encoder_hidden

    def calculate_query(self, batch):
        """

        :param encoder_outputs: B x seq_len x dim
        :param ent_mask: B x max_abs x num_ents x seq_len
        :param outp_ents: B x num_ents (usually 2)
        :return:
        """
        encoder_outputs, ent_mask = batch.encoder_outputs, batch.ent_mask
        # expand
        num_abs = ent_mask.size(1)
        num_ents = ent_mask.size(2)
        seq_len = ent_mask.size(3)
        encoder_outputs = encoder_outputs.unsqueeze(1) # B x 1 x seq_len x dim
        encoder_outputs = encoder_outputs.expand(-1, num_abs, -1, -1).contiguous() # B x num_abs x seq_len x dim
        encoder_outputs = encoder_outputs.view(-1, seq_len, encoder_outputs.size(3)) # (B x num_abs) x seq_len x dim
        ent_mask = ent_mask.view(-1, num_ents, seq_len) # (B x num_abs) x num_ents x seq_len

        query_rep = torch.bmm(ent_mask.float(), encoder_outputs) # (B x num_abs) x num_ents x dim
        query_rep = query_rep.transpose(1,2) # (B x num_abs) x dim x num_ents
        hidden_size = self.model_config.encoder.hidden_dim
        ents = query_rep.size(-1)
        query_reps = []
        for i in range(ents):
            query_reps.append(query_rep.transpose(1,2)[:,i,:].unsqueeze(1))
        query_rep = torch.cat(query_reps, -1)
        return query_rep

    def forward(self, batch, step_batch):
        decoder_inp = step_batch.decoder_inp
        hidden_rep = step_batch.hidden_rep
        query_rep = step_batch.query_rep
        encoder_outputs = batch.encoder_outputs
        _, seq_len, dim = encoder_outputs.size()
        encoder_outputs = encoder_outputs.unsqueeze(1).expand(-1,
            batch.ent_mask.size(1), -1, -1).contiguous().view(-1,seq_len,dim)
        encoder_length = batch.inp_lengths
        # LETS
        check_id_emb(decoder_inp, self.model_config.vocab_size)
        decoder_inp = self.embedding(decoder_inp)
        if self.model_config.only_relation:
            mlp_inp = torch.cat([query_rep.squeeze(1), encoder_outputs[:, -1, :]], -1)
        else:
            lstm_inp = torch.cat([decoder_inp, query_rep], -1)
            mlp_inp, hidden_rep = self.lstm(lstm_inp, hidden_rep)

        outp = self.decoder2vocab(mlp_inp)
        return outp, None, hidden_rep

    def __handle_state__(self, tensor):
        """
        Cut the first dimension in half and concat it to last dimension
        :param tensor:
        :return:
        """
        size_mid = int(tensor.size(0) / 2)
        return torch.cat([tensor[:size_mid], tensor[size_mid:]], -1)

