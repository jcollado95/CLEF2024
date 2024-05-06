import pandas as pd
import random
import argparse
from os.path import join, dirname, basename
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

import sys
sys.path.append('.')

from scorer import evaluate
from format_checker import check_format

random.seed(0)
ROOT_DIR = dirname(dirname(__file__))

def run_positive_baseline(data_fpath, test_fpath, results_fpath, lang):
    train_df = pd.read_csv(data_fpath, dtype=object, encoding="utf-8", sep='\t')
    test_df = pd.read_csv(test_fpath, dtype=object, encoding="utf-8", sep='\t')
    text_head = "tweet_text"
    id_head = "tweet_id"
    if lang == "english":
        text_head = "Text"
        id_head = "Sentence_id"
    with open(results_fpath, "w") as results_file:
            results_file.write("id\tclass_label\trun_id\n")
            for i, line in test_df.iterrows():
                results_file.write("{}\t{}\t{}\n".format(line[id_head], "Yes", "majority"))


def run_majority_baseline(data_fpath, test_fpath, results_fpath, lang):
    train_df = pd.read_csv(data_fpath, dtype=object, encoding="utf-8", sep='\t')
    test_df = pd.read_csv(test_fpath, dtype=object, encoding="utf-8", sep='\t')

    pipeline = DummyClassifier(strategy="most_frequent")

    text_head = "tweet_text"
    id_head = "tweet_id"
    if lang == "english":
        text_head = "Text"
        id_head = "Sentence_id"


    pipeline.fit(train_df[text_head], train_df['class_label'])

    with open(results_fpath, "w") as results_file:
        predicted_distance = pipeline.predict(test_df[text_head])

        results_file.write("id\tclass_label\trun_id\n")

        for i, line in test_df.iterrows():
            label = predicted_distance[i]

            results_file.write("{}\t{}\t{}\n".format(line[id_head], label, "majority"))


def run_random_baseline(data_fpath, results_fpath, lang):
    gold_df = pd.read_csv(data_fpath,  dtype=object, encoding="utf-8", sep='\t')
    label_list= ["Yes", "No"]


    id_head = "tweet_id"
    if lang == "english":
        id_head = "Sentence_id"

    with open(results_fpath, "w") as results_file:
        results_file.write("id\tclass_label\trun_id\n")
        for i, line in gold_df.iterrows():
            results_file.write('{}\t{}\t{}\n'.format(line[id_head],random.choice(label_list), "random"))


def run_ngram_baseline(train_fpath, test_fpath, results_fpath, lang):
    train_df = pd.read_csv(train_fpath, dtype=object, encoding="utf-8", sep='\t')
    test_df = pd.read_csv(test_fpath, dtype=object, encoding="utf-8", sep='\t')

    text_head = "tweet_text"
    id_head = "tweet_id"
    if lang == "english":
        text_head = "Text"
        id_head = "Sentence_id"

    pipeline = Pipeline([
        ('ngrams', TfidfVectorizer(
            ngram_range=(1, 1),
            lowercase=True,
            use_idf=True,
            max_df=0.95, 
            min_df=3,
            max_features=5000
        )),
        ('clf', SVC(C=1, gamma='scale', kernel='linear', random_state=0))
    ])
    pipeline.fit(train_df[text_head], train_df['class_label'])

    with open(results_fpath, "w") as results_file:
        predicted_distance = pipeline.predict(test_df[text_head])
        results_file.write("id\tclass_label\trun_id\n")
        for i, line in test_df.iterrows():
            label = predicted_distance[i]
            results_file.write("{}\t{}\t{}\n".format(line[id_head], label, "ngram"))


def run_baselines(train_fpath, test_fpath, lang):
    positive_baseline_fpath = join(
        ROOT_DIR,
        f'data/positive_baseline_{basename(test_fpath)}'
    )
    run_positive_baseline(train_fpath, test_fpath, positive_baseline_fpath, lang)

    if check_format(positive_baseline_fpath):
        acc, precision, recall, f1 = evaluate(test_fpath, positive_baseline_fpath, lang)
        print(f"Positive Baseline for {lang} metrics (positive class): Acc: {acc}, P: {precision}, R: {recall}, F1: {f1}")

    majority_baseline_fpath = join(ROOT_DIR,
                                 f'data/majority_baseline_{basename(test_fpath)}')
    run_majority_baseline(train_fpath, test_fpath, majority_baseline_fpath, lang)

    if check_format(majority_baseline_fpath):
        acc, precision, recall, f1 = evaluate(test_fpath, majority_baseline_fpath, lang)
        print(f"Majority Baseline for {lang} metrics (positive class): Acc: {acc}, P: {precision}, R: {recall}, F1: {f1}")


    random_baseline_fpath = join(ROOT_DIR, f'data/random_baseline_{basename(test_fpath)}')
    run_random_baseline(test_fpath, random_baseline_fpath, lang)

    if check_format(random_baseline_fpath):
        acc, precision, recall, f1 = evaluate(test_fpath, random_baseline_fpath, lang)
        print(f"Random Baseline for {lang} metrics (positive class): Acc: {acc}, P: {precision}, R: {recall}, F1: {f1}")

    ngram_baseline_fpath = join(ROOT_DIR, f'data/ngram_baseline_{basename(test_fpath)}')
    run_ngram_baseline(train_fpath, test_fpath, ngram_baseline_fpath, lang)
    if check_format(ngram_baseline_fpath):
        acc, precision, recall, f1 = evaluate(test_fpath, ngram_baseline_fpath, lang)
        print(f"Ngram Baseline for {lang} metrics (positive class): Acc: {acc}, P: {precision}, R: {recall}, F1: {f1}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file-path", "-t", required=True, type=str,
                        help="The absolute path to the training data")
    parser.add_argument("--dev-file-path", "-d", required=True, type=str,
                        help="The absolute path to the dev data")
    parser.add_argument("--lang", "-l", required=True, type=str,
                        choices=['arabic', 'english', 'dutch', 'spanish'],
                        help="The language of the task")

    args = parser.parse_args()
    run_baselines(args.train_file_path, args.dev_file_path, args.lang)


    # python3 src/baseline.py --train-file-path=data/CT24_checkworthy_english/CT24_checkworthy_english_train.tsv --dev-file-path=data/CT24_checkworthy_english/CT24_checkworthy_english_dev-test.tsv -l english