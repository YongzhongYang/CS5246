from torch.utils.data import Dataset
from torchtext import data
import numpy as np

class BertLstmDataset(Dataset):
    def __init__(self, all_input_ids, all_input_mask, all_segment_ids, all_label_ids,
                 lstm_train_feas):
        self.all_input_ids = all_input_ids
        self.all_input_mask = all_input_mask
        self.all_segment_ids = all_segment_ids
        self.all_label_ids = all_label_ids
        self.lstm_train_feas = lstm_train_feas

        assert(all_label_ids.size()[0] == all_input_mask.size()[0] and
               all_label_ids.size()[0] == all_segment_ids.size()[0] and
               all_input_ids.size()[0] == all_label_ids.size()[0] and
               all_input_ids.size()[0] == len(lstm_train_feas))

    def __len__(self):
        return self.all_input_ids.size()[0]

    def __getitem__(self, idx):
        return self.all_input_ids[idx], self.all_input_mask[idx], self.all_segment_ids[idx], \
               self.all_label_ids[idx], self.lstm_train_feas[idx]


def load_embeddings(lstm_opt):
    text_field = data.Field(lower=True, fix_length=lstm_opt['max_len'])
    label_field = data.Field(sequential=False)
    train, dev, test = data.TabularDataset.splits(path=lstm_opt['data_base'],
                                                  train="train.tsv",
                                                  validation="dev.tsv",
                                                  test="test.tsv",
                                                  format="tsv",
                                                  fields=[('text', text_field), ('label', label_field)])
    text_field.build_vocab(train, test, dev)
    text_field.vocab.load_vectors(lstm_opt['embedding_type'])
    return text_field, label_field


def sample_data(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, lstm_train_sent):
    total_len = len(lstm_train_sent)
    samples_len = total_len // 10
    idx = np.random.choice(np.arange(total_len), samples_len, replace=False)
    return all_input_ids[idx], all_input_mask[idx], all_segment_ids[idx], all_label_ids[idx], np.array(lstm_train_sent)[idx]
