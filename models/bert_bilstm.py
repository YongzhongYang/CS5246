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


class BERT_BiLSTM(nn.Module):
    def __init__(self, lstm_opt,bert_opt, n_filters,filter_sizes,output_dim,dropout, variant='bert-base-uncased'):
        super(BERT_BiLSTM, self).__init__()

        # BERT layers
        self.bert = BertModel.from_pretrained(variant)
        self.bert_dropout = nn.Dropout(bert_opt["dropout"])

        # LSTM layers
        self.use_gpu = lstm_opt['use_gpu']
        self.batch_size = lstm_opt['batch_size']
        self.hidden_dim = lstm_opt['hidden_dim']
        self.convs = nn.ModuleList([
                                    nn.Conv1d(in_channels = 300, 
                                              out_channels = n_filters, 
                                              kernel_size = fs)
                                    for fs in filter_sizes
                                    ])
        # text_fields, label_fields = self.load_embeddings(lstm_opt)
        text_fields, label_fields = lstm_opt['text_fields'], lstm_opt['label_fields']
        self.embeddings = nn.Embedding.from_pretrained(text_fields.vocab.vectors)
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)

        # Linear layers
        self.dense = nn.Linear(bert_opt["bert_dim"] + lstm_opt['hidden_dim'] * 2, bert_opt["polarities_dim"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def forward(self, inputs):
        bert_inputs, lstm_inputs = inputs
        text_bert_indices, bert_segments_ids = bert_inputs[0], bert_inputs[1]
        text_bert_indices = text_bert_indices.to(self.device)
        bert_segments_ids = bert_segments_ids.to(self.device)
        lstm_inputs = lstm_inputs.to(self.device)
        _, pooled_output = self.bert(text_bert_indices, bert_segments_ids, output_all_encoded_layers=False)
        pooled_output = self.bert_dropout(pooled_output)
        

        embed = self.embeddings(lstm_inputs).view(len(lstm_inputs), self.batch_size, -1)
        embed=embed.permute(1,0,2)
        conved = [F.relu(conv(embed)) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim = 1))

        y = self.dense(torch.cat((pooled_output, self.fc(cat)), dim=1))
        log_probs = F.log_softmax(y)
        del embed
        del conved
        del pooled
        del text_bert_indices
        del bert_segments_ids
        del pooled_output
        del y
        return log_probs
