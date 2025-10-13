def chunks(data, chunksize):
    from itertools import islice
    iterator = iter(data)
    chunk = list(islice(iterator, chunksize))
    while chunk:
        yield chunk
        chunk = list(islice(iterator, chunksize))


def find_threshold(y_true, similarities, top_thresholds=3, min_f1=0.7):
    thresholds = []
    for threshold in (i / 100 for i in range(100)):
        f1 = get_results(y_true, similarities, threshold, min_f1)
        thresholds.append((f1, threshold))
    return list(zip(*sorted(thresholds, reverse=True)))[-1][:top_thresholds]


def get_results(y_true, similarities, threshold, min_f1):
    from sklearn.metrics import recall_score, precision_score

    predicted = [int(o > threshold) for o in similarities]
    if len(predicted) == 0:
        return 0
    recall_score_res = recall_score(y_true, predicted) or 0
    precision_score_res = precision_score(y_true, predicted) or 0
    if recall_score_res == 0 and precision_score_res == 0:
        f1 = 0
    else:
        f1 = 2 * recall_score_res * precision_score_res / (
                precision_score_res + recall_score_res)
    if f1 >= min_f1:
        print('threshold= ', threshold,
              "\trecall= ", round(recall_score_res, 3),
              "\tprecision= ", round(precision_score_res, 3),
              "\tF1-score", round(f1, 3))
    return f1


def load_data(train_path='dataset_level_3_train_sports.csv',
              test_path='dataset_level_3_test_sports.csv'):
    import pandas as pd
    train_dataset = pd.read_csv(train_path)
    test_dataset = pd.read_csv(test_path)

    train1 = train_dataset['Text_1']
    train2 = train_dataset['Text_2']
    train_y = train_dataset['Result']

    test1 = test_dataset['Text_1']
    test2 = test_dataset['Text_2']
    test_y = test_dataset['Result']

    return train1, train2, train_y, test1, test2, test_y


def parallelizer(fn, iterator, n_jobs=None, verbose=False, desc=None,
                 *args, **kwargs):
    import multiprocessing as mp
    from functools import partial
    from tqdm import tqdm

    n_cpu = mp.cpu_count()
    n_jobs = n_jobs or max(12, n_cpu * 1 // 2) if n_cpu > 12 else n_cpu - 1

    pool = mp.Pool(n_jobs)
    fn = partial(fn, *args, **kwargs)
    iterator = tqdm(iterator, desc=desc) if verbose else iterator
    output = pool.map(fn, iterator)
    pool.close()
    pool.join()
    return output


def set_device(device=0):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
    print(f"Setting GPU device: {device}")


def train_and_test(similarity_function, train_path, test_path, top_thresholds=3,
                   min_f1=0.7):
    from tqdm import tqdm
    train1, train2, train_y, test1, test2, test_y = load_data(train_path,
                                                              test_path)
    print(80 * '-')
    print('Training')
    print(80 * '-')

    train_sentence_pairs = tqdm(zip(train1, train2), total=len(train1))
    train_similarities = similarity_function(train_sentence_pairs)
    best_threshold = find_threshold(train_y, train_similarities, top_thresholds,
                                    min_f1)

    print(80 * '-')
    print('Testing')
    print(80 * '-')
    test_sentence_pairs = tqdm(zip(test1, test2), total=len(test1))
    test_similarities = similarity_function(test_sentence_pairs)
    for t in best_threshold:
        get_results(test_y, test_similarities, t, min_f1=0)
