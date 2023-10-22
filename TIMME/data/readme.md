# TIMME Dataset Documentation

Our dataset is collected from Twitter API in 2019, by March.

## Recent Updates on 7 March 2021

We proide the tweet IDs posted on [Box](https://ucla.box.com/s/47i6p7mkaer1f4ec8c0qcyxj1u0g3z8m).
The [pure text content](https://ucla.box.com/s/nk27vfb26jfhqrvyfv9e8m3dvbz11sq1) grouped by account ID is also provided.

## Relational Dataset (P50, P_20_50)

These datasets are of the same format, corresponding to the four dataset we've used in our paper.

To use our dataset and/or code, please cite [our paper](https://arxiv.org/abs/2006.01321).
```
@article{xiao2020timme,
  title={TIMME: Twitter Ideology-detection via Multi-task Multi-relational Embedding},
  author={Xiao, Zhiping and Song, Weiping and Xu, Haoyan and Ren, Zhicheng and Sun, Yizhou},
  journal={arXiv preprint arXiv:2006.01321},
  year={2020}
}
```

The components in our datasets:
* dict.csv: the ground-truth labels of the politicians.
* friend_list.csv: the relation of A follow B.
* favorite_list.csv: the relation of A like B's tweet.
* mention_list.csv: the relation of A mention B in her/his tweet (retweeting's automatic @ is excluded).
* reply_list.csv: the relation of A replying to B's tweet.
* retweet_list.csv: the relation of A retweet from B. This retweet include retweeting with / without comments. In other places, *retweet with comment* is sometimes referred to as "quote". The reason why we treated them the same is that, from a Twitter user's side these two options are almost the same.
* tweet_features.npz: the file storing the average GloVe embeddings of the words in each user's tweets (all the tweets in our record). Sample usage:

```python
import numpy as np
features_file = "tweet_features.npz"
loaded = np.load(features_file)
print(loaded["tweets_average"][0,:])
```

* features.npz: the file storing the average GloVe embeddings of the words in each user's account **description** or **status** fields. Sample usage:

```python
import numpy as np
features_file = "features.npz"
loaded = np.load(features_file)
print(loaded['description'][0,:])
print(loaded['status'][0,:])
```

* all_twitter_ids.csv: it is useful only when you load the features, as the features in *tweet_features.npz* or *features.npz* are stored as numpy matrix, and each row corresponds to a user. Those user's twitter ids are as listed here in all_twitter_ids.csv.

All relations are in the format of ```<from>	<to>	<count>```, where *from* is where this relation starts, *to* is whom involved on the other side, *count* is how many times these two person have this relationship in between in the time window we observe. (**note:** all relations are directed --- from A to B and from B to A are treated differently.)

Something to note is that, in order to safely contain the politicians' names and probably some other special characters, we decided to **use ```\t``` as our separator**, for almost all the data files we use.

## Additional Labels

Additional labels are extracted according to the user profile information's **description** field, done by Zhicheng, manually fitering out the negative references after automatically select a bunch of potential candidates.

The additional labels we have are stored under [additional_labels](./additional_labels) folder.

For mored details, please visit [TIMME-additional-labels](https://github.com/franklinnwren/TIMME-additional-labels).
