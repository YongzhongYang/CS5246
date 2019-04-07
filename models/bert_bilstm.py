# -*- coding: utf-8 -*-
# file: BERT_SPC.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2019. All Rights Reserved.
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


# from layers.squeeze_embedding import SqueezeEmbedding
from pytorch_pretrained_bert import BertModel, BertForSequenceClassification, BertForNextSentencePrediction, \
    BertForPreTraining

#elmo
from allennlp.modules.elmo import Elmo

options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

class BERT_BiLSTM(nn.Module):
    def __init__(self, lstm_opt, bert_opt, variant='bert-base-uncased'):
        super(BERT_BiLSTM, self).__init__()

        # BERT layers
        self.bert = BertModel.from_pretrained(variant)
        self.bert_dropout = nn.Dropout(bert_opt["dropout"])
        self.elmo = Elmo(options_file, weight_file, 2, dropout=0)
        # LSTM layers
        self.use_gpu = lstm_opt['use_gpu']
        self.batch_size = lstm_opt['batch_size']
        self.hidden_dim = lstm_opt['hidden_dim']
        self.lstm_dropout = lstm_opt['dropout']
        # text_fields, label_fields = self.load_embeddings(lstm_opt)
        #text_fields, label_fields = lstm_opt['text_fields'], lstm_opt['label_fields']
        #self.embeddings = nn.Embedding.from_pretrained(text_fields.vocab.vectors)
        #self.bilstm = nn.LSTM(input_size=text_fields.vocab.vectors.size()[1],
        self.bilstm = nn.LSTM(input_size=1024,
                              hidden_size=lstm_opt['hidden_dim'], bidirectional=True)
        self.hidden = self.init_hidden()

        # Linear layers
        self.dense = nn.Linear(bert_opt["bert_dim"] + lstm_opt['hidden_dim'] * 2, bert_opt["polarities_dim"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def init_hidden(self):
        # first is the hidden h
        # second is the cell c
        if self.use_gpu:
             return (Variable(torch.zeros(2, self.batch_size, self.hidden_dim).cuda()),
                     Variable(torch.zeros(2, self.batch_size, self.hidden_dim).cuda()))
        else:
             return (Variable(torch.zeros(2, self.batch_size, self.hidden_dim)),
                     Variable(torch.zeros(2, self.batch_size, self.hidden_dim)))

    def forward(self, inputs):
        bert_inputs, lstm_inputs = inputs
        text_bert_indices, bert_segments_ids = bert_inputs[0], bert_inputs[1]
        text_bert_indices = text_bert_indices.to(self.device)
        bert_segments_ids = bert_segments_ids.to(self.device)
        lstm_inputs = lstm_inputs.to(self.device)
        _, pooled_output = self.bert(text_bert_indices, bert_segments_ids, output_all_encoded_layers=False)
        pooled_output = self.bert_dropout(pooled_output)

        #lstm_x = self.embeddings(lstm_inputs).view(len(lstm_inputs), self.batch_size, -1)
        lstm_x = self.elmo(lstm_inputs).permute(0,2,1)
        lstm_y, self.hidden = self.bilstm(lstm_x, self.hidden)

        y = self.dense(torch.cat((pooled_output, lstm_y[-1]), dim=1))
        log_probs = F.log_softmax(y)
        del text_bert_indices
        del bert_segments_ids
        del lstm_inputs
        del pooled_output
        del lstm_x
        del lstm_y
        del y
        return log_probs
