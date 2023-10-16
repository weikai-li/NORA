import numpy as np
import torch
from utils import multi_relation_load
from sampler import NaiveSampler as TIMMESampler


def cycle_data_split(train_idx, val_idx, test_idx, cycle):
    all_idx = torch.cat([train_idx, val_idx, test_idx])
    offset = len(all_idx) * (cycle) / 5
    offset = int(offset)
    train_idx1 = all_idx[offset: min(len(train_idx)+offset, len(all_idx))]
    train_idx2 = all_idx[max(0, offset-len(all_idx)): max(0, len(train_idx)+offset-len(all_idx))]
    train_idx = torch.cat([train_idx1, train_idx2])
    val_idx1 = all_idx[min(len(train_idx)+offset, len(all_idx)): min(len(train_idx)+len(val_idx)+offset, len(all_idx))]
    val_idx2 = all_idx[max(0, len(train_idx)+offset-len(all_idx)): max(0, len(train_idx)+len(val_idx)+offset-len(all_idx))]
    val_idx = torch.cat([val_idx1, val_idx2])
    test_idx1 = all_idx[min(len(train_idx)+len(val_idx)+offset, len(all_idx)): len(all_idx)]
    test_idx2 = all_idx[max(0, offset-len(test_idx)): max(0, offset)]
    test_idx = torch.cat([test_idx1, test_idx2])

    assert len(train_idx) + len(val_idx) + len(test_idx) == len(all_idx)
    try:
        train_set = set([i.item() for i in train_idx])
        val_set = set([i.item() for i in val_idx])
        test_set = set([i.item() for i in test_idx])
        assert len(train_set & val_set) == 0
        assert len(train_set & test_set) == 0
        assert len(val_set & test_set) == 0
    except:
        pass
    return train_idx, val_idx, test_idx


def sample(dataset, cycle):
    adjs_ori, features_ori, labels_info, trainable, mask, link_info, (label_map, all_id_list) = \
        multi_relation_load(f'../data/{dataset}', 
        files=['retweet_list.csv', 'mention_list.csv', 'friend_list.csv', 'reply_list.csv', 'favorite_list.csv'], 
        feature_data='one_hot', feature_file=None, freeze_feature=False, split_links=True,
        additional_labels_files=["../data/additional_labels/new_dict_cleaned.csv"])
    
    train_link, val_link, test_link = link_info
    if cycle != 0:
        new_test_link = []
        for i in range(5):
            train_link_tmp, val_link_tmp, test_link_tmp = cycle_data_split(
                torch.tensor(train_link[i].T), torch.tensor(val_link[i].T), 
                torch.tensor(test_link[i].T), cycle
            )
            new_test_link.append(np.array(test_link_tmp).T)
        test_link = new_test_link

    sampler = TIMMESampler(adjs_ori[0].size()[0], test_link, n_batches=1, negative_rate=1.5, 
        report_interval=0, epochs=1, separate_relations=True)
    cnt = 0
    for batch_id, triplets, labels, relation_indexes, _ in sampler.batch_generator():
        cnt += 1
        for i in range(5):
            np.save(f'../data/{dataset}/sampled_triplets_{cycle}_{i}.npy', triplets[i])
    assert cnt == 1


if __name__ == '__main__':
    sample('P50', 0)
    sample('P50', 1)
    sample('P50', 2)
    sample('P50', 3)
    sample('P50', 4)
    sample('P_20_50', 0)
    sample('P_20_50', 1)
    sample('P_20_50', 2)
    sample('P_20_50', 3)
    sample('P_20_50', 4)
