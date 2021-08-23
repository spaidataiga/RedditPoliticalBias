
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function, Variable

import math

class Encoder(nn.Module):
  def __init__(self, embedding_dim, hidden_dim, nlayers=1, dropout=0.,
               bidirectional=True, rnn_type='GRU'):
    super(Encoder, self).__init__()
    self.bidirectional = bidirectional
    assert rnn_type in ['LSTM', 'GRU'], 'Use one of the following: {}'.format(str(RNNS))
    rnn_cell = getattr(nn, rnn_type) # fetch constructor from torch.nn, cleaner than if
    self.rnn = rnn_cell(embedding_dim, hidden_dim, nlayers,
                        dropout=dropout, bidirectional=bidirectional)

  def forward(self, input, hidden=None):
    self.rnn.flatten_parameters()
    return self.rnn(input, hidden)

class BahdanauAttention(nn.Module):
  def __init__(self, hidden_dim, attn_dim):
    super(BahdanauAttention, self).__init__()
    self.linear = nn.Linear(hidden_dim, attn_dim)
    self.linear2 = nn.Linear(attn_dim, 1)

  def forward(self, hidden, mask=None):
    # hidden = [TxBxH]
    # mask = [TxB]
    # Outputs = a:[TxB], lin_comb:[BxV]

    # print (hidden.size())
    # Here we assume q_dim == k_dim (dot product attention)
    hidden = hidden.transpose(0,1) # [TxBxH] -> [BxTxH]
    energy = self.linear(hidden) # [BxTxH] -> [BxTxA]
    energy = F.tanh(energy)
    energy = self.linear2(energy) # [BxTxA] -> [BxTx1]
    energy = F.softmax(energy, dim=1) # scale, normalize

    # print (energy.size())
    if mask is not None:
      mask = mask.transpose(0, 1).unsqueeze(2)
      # print (mask.size())
      energy = energy * mask
      # print (energy.size())
      Z = energy.sum(dim=1, keepdim=True) #[BxTx1] -> [Bx1x1]
      # print (Z.size())
      # input()
      energy = energy/Z #renormalize

    energy = energy.transpose(1, 2) # [BxTx1] -> [Bx1xT]
    # hidden = hidden.transpose(0,1) # [TxBxH] -> [BxTxH]
    linear_combination = torch.bmm(energy, hidden).squeeze(1) #[Bx1xT]x[BxTxH] -> [BxH]
    return energy, linear_combination

class Classifier_GANLike(nn.Module):
  def __init__(self, embedding, encoder, attention, hidden_dim, num_classes=10):
    super(Classifier_GANLike, self).__init__()
    # num_classes=2
    self.embedding = embedding
    self.encoder = encoder
    self.attention = attention
    self.decoder = nn.Linear(hidden_dim, num_classes)

    size = 0
    for p in self.parameters():
      size += p.nelement()
    print('Total param size: {}'.format(size))

  def forward(self, input, padding_mask=None, rationale_mask = None):
    if rationale_mask is not None:
        x_embeds = self.embedding(input.squeeze(1))
        x_embeds = x_embeds * rationale_mask.unsqueeze(-1)
    else:
        x_embeds = self.embedding(input)
    outputs, hidden = self.encoder(x_embeds)
    if isinstance(hidden, tuple): # LSTM
      hidden = hidden[1] # take the cell state

    if self.encoder.bidirectional: # need to concat the last 2 hidden layers
      hidden = torch.cat([hidden[-1], hidden[-2]], dim=1)
    else:
      hidden = hidden[-1]

    # max across T?
    # Other options (work worse on a few tests):
    # linear_combination, _ = torch.max(outputs, 0)
    # linear_combination = torch.mean(outputs, 0)

    energy, linear_combination = self.attention(outputs, padding_mask)
    logits = self.decoder(linear_combination)

    # if gradreverse:
    #   reverse_linear_comb = ReverseLayerF.apply(linear_combination, alpha)
    #   topic_logprobs = self.topic_decoder(reverse_linear_comb)
    # else:
    #   topic_logprobs = self.topic_decoder(linear_combination)
    return logits, energy, linear_combination
