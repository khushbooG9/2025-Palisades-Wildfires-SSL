import torch

class Dataset_tracked(torch.utils.data.Dataset):
    def __init__(self, text_list, labels, idxes, tokenizer, max_seq_len=128, labeled=True):
        self.text_list = text_list
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.labeled = labeled
        self.idxes = idxes
        self.weights = [1] * len(self.text_list)

    def __getitem__(self, idx):
        tok = self.tokenizer(
            self.text_list[idx], padding='max_length', max_length=self.max_seq_len, truncation=True)
        item = {key: torch.tensor(tok[key]) for key in tok}
        if self.labeled == True:
            item['lbl'] = torch.tensor(self.labels[idx], dtype=torch.long)
        item['idx'] = self.idxes[idx]
        item['weights'] = self.weights[idx]
        return item

    def __len__(self):
        return len(self.text_list)


    def get_subset_dataset(self, idxs): 
        text_lists, label_lists, id_lists = [], [], []
        for text, label, id in zip(self.text_list, self.labels, self.idxes):
            for i in idxs:
                if i==id:
                    text_lists.append(text)
                    label_lists.append(label)
                    id_lists.append(id)

        return CustomDataset_tracked(text_lists, label_lists, id_lists, self.tokenizer, self.max_seq_len, self.labeled)