# Utilities for encoding
import torch

class EncoderUtils():
    """
    Utility to pack sentences,
    then unpack
    """

    def __init__(self):
        self.sorted_idx = None
        self.batch_size = None
        self.num_sentences = None
        self.num_words = None
        self.data_indices = None
        self.data_lengths_sorted = None
        self.data_mask = None

    def pack(self, data, sent_lengths):
        batch_size, num_sentences, num_words = data.size()
        # view data into sentence
        data = data.view((batch_size * num_sentences), -1)  # (B x sent) x words
        data_lengths = [t for sent in sent_lengths for t in sent]  # (B x sent)
        assert data.size(0) == len(data_lengths)
        # Sort the sentences
        data_lengths_tensor = torch.tensor(data_lengths)
        data_lengths_sorted, sorted_idx = data_lengths_tensor.sort(descending=True)
        data = data[sorted_idx]
        data_indices = torch.tensor([idx for idx, l in enumerate(data_lengths) if l > 0])
        data_indices = data_indices.to(data.device)
        data_mask = torch.tensor([1 if l > 0 else 0 for l in data_lengths]).byte()
        data_mask = data_mask.to(data.device)
        self.sorted_idx = sorted_idx
        self.batch_size = batch_size
        self.num_sentences = num_sentences
        self.num_words = num_words
        self.data_indices = data_indices
        self.data_lengths_sorted = data_lengths_sorted
        self.data_mask = data_mask
        return data

    def unpack(self, data):
        # get the sentences back from sorting
        sorted_idx = self.sorted_idx.to(data.device)
        data_state = torch.zeros_like(data).scatter_(
            0, sorted_idx.unsqueeze(1).unsqueeze(1)
                .expand(-1, data.size(1), data.size(2)), data)

        # change the view back to num_sentence and words
        data_state = data_state.view(self.batch_size,
                                     self.num_sentences,
                                     self.num_words,
                                     data_state.size(-1))
        data_state = data_state.contiguous()  # batch x sents x words x dim
        return data_state