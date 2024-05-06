from scorer import evaluate
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold-file-path", "-g", required=True, type=str,
                        help="The absolute path to the gold data")
    parser.add_argument("--pred-file-path", "-p", required=True, type=str,
                        help="The absolute path to the pred data")
    parser.add_argument("--lang", "-l", type=str,
                        choices=['arabic', 'english', 'dutch', 'spanish'],
                        help="The language of the task",
                        default="english")                        

    args = parser.parse_args()
    acc, precision, recall, f1 = evaluate(
        args.gold_file_path, 
        args.pred_file_path,
        "english"
    )

    print(f"Metrics (positive class): Acc: {acc}, P: {precision}, R: {recall}, F1: {f1}")

# python3 src/evaluate_experiment.py --gold-file-path=data/CT24_checkworthy_english/CT24_checkworthy_english_dev-test.tsv --pred-file-path=data/gpt-fs-no-ctx_CT24_checkworthy_english_dev-test.tsv