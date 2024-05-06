import logging
import argparse

import sys
sys.path.append('.')
from format_checker import check_format
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

"""
Scoring of Task 1 with the metrics accuracy, and precision, recall, and f1 for positive class.
"""

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)


MAIN_THRESHOLDS = [1, 3, 5, 10, 20, 50]

def _read_gold_and_pred(gold_fpath, pred_fpath, lang):
    """
    Read gold and predicted data.
    :param gold_fpath: the original annotated gold file, where the last 4th column contains the labels.
    :param pred_fpath: a file with line_number and score at each line.
    :param lang: If en, read debate sentence, otherwise read tweets
    :return: {line_number:label} dict; list with (line_number, score) tuples.
    """

    logging.info("Reading gold labels from file {}".format(gold_fpath))

    gold_labels = {}
    if lang == "english":
        with open(gold_fpath, encoding='utf-8') as gold_f:
            next(gold_f)
            for line_res in gold_f:
                (id, text, label) = line_res.strip().split('\t')  # process the line from the res file
                gold_labels[str(id)] = label

    else:
        with open(gold_fpath, encoding='utf-8') as gold_f:
            next(gold_f)
            for line_res in gold_f:
                (tweet_id, tweet_url, tweet_text, label) = line_res.strip().split('\t')  # process the line from the res file
                gold_labels[str(tweet_id)] = label

    logging.info('Reading predicted labels from file {}'.format(pred_fpath))

    line_score = []
    with open(pred_fpath) as pred_f:
        next(pred_f) # Skip header
        for line in pred_f:
            id, pred_label, run_id  = line.split('\t')
            id = str(id.strip())
            pred_label = pred_label.strip()

            if id not in gold_labels:
                logging.error('No such id: {} in gold file!'.format(id))
                quit()
            line_score.append((id, pred_label))


    if len(set(gold_labels).difference([tup[0] for tup in line_score])) != 0:
        logging.error('The predictions do not match the lines from the gold file - missing or extra line_no')
        raise ValueError('The predictions do not match the lines from the gold file - missing or extra line_no')

    return gold_labels, line_score


def evaluate(gold_fpath, pred_fpath, lang="arabic"):
    """
    Evaluates the predicted line rankings w.r.t. a gold file.
    Metrics are: Average Precision, R-Pr, Reciprocal Rank, Precision@N
    :param gold_fpath: the original annotated gold file, where the last 4th column contains the labels.
    :param pred_fpath: a file with line_number at each line, where the list is ordered by check-worthiness.
    :param lang: arabic, english, dutch, or spanish.
    """

    gold_labels_dict, pred_labels_dict = _read_gold_and_pred(gold_fpath, pred_fpath, lang)
    gold_labels=[]
    pred_labels=[]
    for t_id, label in sorted(gold_labels_dict.items()):
        gold_labels.append(label)
    for t_id, label in sorted(pred_labels_dict):
        pred_labels.append(label)

    acc = accuracy_score(gold_labels, pred_labels)
    precision = precision_score(gold_labels, pred_labels, pos_label='Yes',average='binary')
    recall = recall_score(gold_labels, pred_labels, pos_label='Yes',average='binary')
    f1 = f1_score(gold_labels, pred_labels, pos_label='Yes',average='binary')

    return acc, precision, recall, f1


def validate_files(pred_file):
    if not check_format(pred_file):
        logging.error('Bad format for pred file {}. Cannot score.'.format(pred_file))
        return False
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold-file-path", "-g", required=True, type=str,
                        help="Path to file with gold annotations.")
    parser.add_argument("--pred-file-path", "-p", required=True, type=str,
                        help="Path to file with predict class per tweet.")
    parser.add_argument("--lang", "-l", required=True,
                        choices=['arabic', 'english', 'dutch', 'spanish'],
                        help="The language you want to score its runs.")
    args = parser.parse_args()

    pred_file = args.pred_file_path
    gold_file = args.gold_file_path
    lang = args.lang

    if validate_files(pred_file):
        logging.info(f"Started evaluating results for task-1_{lang} ...")
        overall_precisions = [0.0] * len(MAIN_THRESHOLDS)

        acc, precision, recall, f1 = evaluate(gold_file, pred_file, lang=lang)
        print("acc: {}, P:{}, R:{}, F1:{}".format(acc, precision, recall, f1))

