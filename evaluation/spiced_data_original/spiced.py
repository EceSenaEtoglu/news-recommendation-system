import pandas as pd
from itertools import combinations
from simhash import Simhash
import multiprocessing as mp
from utils import parallelizer
import warnings

CPU = mp.cpu_count()
MAX_JOBS = 12
N_JOBS = max(MAX_JOBS, CPU * 1 // 2) if MAX_JOBS > 12 else CPU - 1

PATH = 'data/spiced.csv'
SEED = 42
TRAIN_SIZE = 0.7
N_HARD = 3000


def shuffle_df(df, seed=SEED):
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def get_similarity(row):
    return Simhash(row['Text_1']).distance(Simhash(row['Text_2']))


def load_train_test_data(path=PATH, train_size=TRAIN_SIZE, seed=SEED):
    df = pd.read_csv(path)
    df = shuffle_df(df, seed=seed)

    train = []
    test = []
    for t in set(df['Type']):
        df_t = df[df['Type'] == t]
        split_idx = int(train_size * len(df_t))
        train.append(df_t[:split_idx].reset_index(drop=True))
        test.append(df_t[split_idx:].reset_index(drop=True))

    train = pd.concat(train, axis=0, ignore_index=True)
    test = pd.concat(test, axis=0, ignore_index=True)
    return train, test


def load_combined(mode='train', path=PATH, train_size=TRAIN_SIZE, seed=SEED):
    train, test = load_train_test_data(path=path, seed=seed, train_size=train_size)
    if mode == 'train':
        df = train
    elif mode == 'test':
        df = test
    else:
        raise ValueError("Mode must be within ['train', 'test']")

    df = df.reset_index().rename(columns={'index': 'pair_id'})
    news = []
    for _, pair in df.iterrows():
        news.append((pair['text_1'], pair['URL_1'], pair['Type'], pair['pair_id']))
        news.append((pair['text_2'], pair['URL_2'], pair['Type'], pair['pair_id']))

    combined = pd.DataFrame(news1 + news2 for news1, news2 in combinations(news, 2))
    combined.columns = ['Text_1', 'URL_1', 'Type_1', 'Pair_ID_1', 'Text_2', 'URL_2', 'Type_2', 'Pair_ID_2']
    combined['Result'] = (combined['Pair_ID_1'] == combined['Pair_ID_2']).astype(int)
    combined = shuffle_df(combined, seed=seed)
    return combined


def load_intertopic(mode='train', path=PATH, train_size=TRAIN_SIZE, seed=SEED):
    df = load_combined(mode=mode, path=path, train_size=train_size, seed=seed)
    f = (df['Pair_ID_1'] == df['Pair_ID_2']) | ((df['Pair_ID_1'] != df['Pair_ID_2']) & (df['Type_1'] != df['Type_2']))
    intertopic = shuffle_df(df[f], seed=seed)
    return intertopic


def load_intratopic_and_hard_examples(mode='train', path=PATH, train_size=TRAIN_SIZE, seed=SEED, n_hard=N_HARD):
    df = load_combined(mode=mode, path=path, train_size=train_size, seed=seed)
    f = (df['Type_1'] == df['Type_2'])
    intratopic_all = df[f]

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=pd.core.common.SettingWithCopyWarning)
        similarities = parallelizer(get_similarity, [o[1] for o in intratopic_all.iterrows()], verbose=True)
        intratopic_all['Similarity'] = list(similarities)

    positives = intratopic_all[intratopic_all['Result'] == 1]
    negatives = intratopic_all[intratopic_all['Result'] == 0]

    if mode == 'train':
        split_idx = int(train_size * n_hard)
    elif mode == 'test':
        split_idx = int((1 - train_size) * n_hard)
    else:
        raise ValueError("Mode must be within ['train', 'test']")

    intratopic_per_topic = {}
    hard_per_topic = {}
    for topic in set(intratopic_all['Type_1']):
        positives_t = positives[positives['Type_1'] == topic]
        negatives_t = negatives[negatives['Type_1'] == topic].sort_values(by='Similarity', ascending=True)
        intratopic = shuffle_df(pd.concat([positives_t, negatives_t[split_idx:]], axis=0, ignore_index=True), seed=seed)
        hard = shuffle_df(pd.concat([positives_t, negatives_t[:split_idx]], axis=0, ignore_index=True), seed=seed)
        intratopic_per_topic[topic] = intratopic
        hard_per_topic[topic] = hard

    return intratopic_per_topic, hard_per_topic


if __name__ == '__main__':
    for mode in ['train', 'test']:
        combined = load_combined(mode)
        combined.to_csv('data/dataset_combined_' + mode + '.csv', index=False)
        intertopic = load_intertopic(mode)
        intertopic.to_csv('data/dataset_intertopic_' + mode + '.csv', index=False)
        intratopic, hard = load_intratopic_and_hard_examples(mode)
        for topic in intratopic:
            intratopic[topic].to_csv(f'data/dataset_intratopic_{topic}_{mode}.csv', index=False)
            hard[topic].to_csv(f'data/dataset_hard_{topic}_{mode}.csv', index=False)
