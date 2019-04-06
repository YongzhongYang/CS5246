import os
import math
import sys
import gzip
import re
import torch
import time
import torch.nn as nn
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torchtext import data,datasets
from string import punctuation



#Customize this string to remove punctuation

#Reproduction Seed
SEED = 1234
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

def load_document(file_name):
    texts = []
    labels = []
    with open(file_name) as f:
        next(f)
        for line in f:
            line = line.split('\t')
            text = line[0].lower()
            texts.append(text)
            labels.append(float(line[1].strip()))
        f.close()
    

    return texts, labels

def load_document_test(file_name):
    texts = []
    labels = []
    with open(file_name) as f:
        next(f)
        for line in f:
            line = line.split('\t')
            text = line[1].lower()
            texts.append(text)
            labels.append(float(line[0].strip()))
        f.close()
    return texts, labels


def get_dataset(df_data, text_field, label_field, test=False):
    fields = [("id", None), # we won't be needing the id, so we pass in None as the field
                 ("text", text_field), ("label", label_field)]       
    examples = []

    if test:
        for text in tqdm(df_data['text']):
            examples.append(data.Example.fromlist([None, text, None], fields))
    else:
        for text, label in tqdm(zip(df_data['text'], df_data['label'])):
            examples.append(data.Example.fromlist([None, text, label], fields))
    return examples, fields

class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, 
                 dropout, pad_idx):
        
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        
        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1, 
                                              out_channels = n_filters, 
                                              kernel_size = (fs, embedding_dim)) 
                                    for fs in filter_sizes
                                    ])
        
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        
        #text = [sent len, batch size]
        
        text = text.permute(1, 0)
                
        #text = [batch size, sent len]
        
        embedded = self.embedding(text)
                
        #embedded = [batch size, sent len, emb dim]
        
        embedded = embedded.unsqueeze(1)
        
        #embedded = [batch size, 1, sent len, emb dim]
        
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
            
        #conv_n = [batch size, n_filters, sent len - filter_sizes[n]]
        
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        
        #pooled_n = [batch size, n_filters]
        
        cat = self.dropout(torch.cat(pooled, dim = 1))

        #cat = [batch size, n_filters * len(filter_sizes)]
            
        return self.fc(cat)

def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum() / len(correct)
    return acc

def train(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:
        
        optimizer.zero_grad()
        
        predictions = model(batch.text).squeeze(1)
        
        loss = criterion(predictions, batch.label)
        
        acc = binary_accuracy(predictions, batch.label)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:

            predictions = model(batch.text).squeeze(1)
            
            loss = criterion(predictions, batch.label)
            
            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

if __name__ == "__main__":
    text_train, labels_train = load_document('./SST-2/train.tsv')
    train_data = pd.DataFrame({'text': text_train, 'label' :labels_train})

    #Define two data field
    TEXT = data.Field()
    LABEL = data.LabelField(dtype = torch.float)
    # construct training dataset examples
    train_examples, train_fields = get_dataset(train_data, TEXT, LABEL)
    train_dataset = data.Dataset(train_examples, train_fields)
    TEXT.build_vocab(train_dataset, vectors = "glove.6B.300d")
    LABEL.build_vocab(train_dataset)

    BATCH_SIZE = 64

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_iterator = data.BucketIterator((train_dataset), batch_size = BATCH_SIZE, device = device)
    INPUT_DIM = len(TEXT.vocab)
    EMBEDDING_DIM = 300
    N_FILTERS = 500
    FILTER_SIZES = [2,3,4,5]
    OUTPUT_DIM = 1
    DROPOUT = 0.5
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]


    model = CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX)
    pretrained_embeddings = TEXT.vocab.vectors

    model.embedding.weight.data.copy_(pretrained_embeddings)
    UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

    model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
    model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
    optimizer = optim.Adam(model.parameters())

    criterion = nn.BCEWithLogitsLoss()

    model = model.to(device)
    criterion = criterion.to(device)
    N_EPOCHS = 20

    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):

        start_time = time.time()
    
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    
        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    text_test, labels_test = load_document('./SST-2/dev.tsv')
    test_data = pd.DataFrame({'text': text_test, 'label' :labels_test})

    # construct training dataset examples
    test_examples, test_fields = get_dataset(test_data, TEXT, LABEL)


    test_dataset = data.Dataset(test_examples, test_fields)
    test_iterator = data.BucketIterator((test_dataset), batch_size = BATCH_SIZE, device = device)
    
    test_loss, test_acc = evaluate(model, test_iterator, criterion)

    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')