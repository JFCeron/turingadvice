import os
import sys
import json
from time import time
from tqdm import tqdm
from absl import flags
from datetime import datetime

from data.assertions import question_is_valid, answer_is_valid, answer_pair_is_valid
from data.to_tfrecord_t5 import encoder, _trim_to_desired_length, _fix_reddit_text
from reward.comparative.data import SELFTEXT_DESIRED_LEN, LOCAL_TSV_PATH

PARAMS_OUT_PATH = os.path.join(os.path.dirname(__file__), "{dataset_id}/params.json")

def _define_flags():
    flags.DEFINE_string(
        name="jsonl_path",
        default="data/redditadvice2019.jsonl",
        help="Dataset generated by create_redditadvice_2019.py"
    )
    flags.DEFINE_integer(
        name="max_time_diff",
        default=None,
        help="Maximum time difference between answer pairs in seconds"
    )
    flags.DEFINE_float(
        name="max_len_ratio",
        default=None,
        help="Maximum length ratio between longest and shortest answer in a pair"
    )
    flags.DEFINE_float(
        name="min_score_ratio",
        default=None,
        help="Minimum score ratio between highest and lowest scoring answers in pair"
    )
    return flags.FLAGS

def to_tsv_line(question, ans1, ans2):
    """
    Creates a tsv line from an answer pair.
    - The latter answer in the line is the one with the higher score.
    - Only question selftext is trimmed to <=1250 tokens, as per Rowan's 
      preprocessing of RedditAdvice2019.

    Args:
    question : dict
        A line from the dataset resulting from data.create_redditadvice_2019.
    ans1 : dict
        An element of question["good_comments"].
    ans2 : dict
        An element of question["good_comments"].
    
    Returns:
    line : str
        A tab-separated line with fields [subreddit, date, title, selftext,
        ans1.body, ans2.body].
    """
    dt_date = datetime.utcfromtimestamp(question["created_utc"])
    str_date = [
        'January', 'February', 'March', 'April', 'May', 'June', 'July',
        'August', 'September', 'October', 'November', 'December'
        ][dt_date.month - 1] \
        + ' {}, {}'.format(dt_date.day, dt_date.year)
    # Sort the answers by score
    if ans1["score"] > ans2["score"]:
        ans1, ans2 = ans2, ans1
    inputs = "Subreddit: {} Date: {} Title: {} Selftext: {}".format(
        _fix_reddit_text(question["subreddit"]),
        _fix_reddit_text(str_date),
        _fix_reddit_text(question["title"]),
        _fix_reddit_text(_trim_to_desired_length(
            encoder,
            question["selftext"],
            desired_len=SELFTEXT_DESIRED_LEN
        ))
    )
    return "\t".join([
        inputs,
        _fix_reddit_text(ans1["body"]),
        _fix_reddit_text(ans2["body"])
    ])

if __name__ == "__main__":
    FLAGS = _define_flags()
    FLAGS(sys.argv)
    # Create directory to store result dataset
    dataset_id = int(time())
    out_dir = os.path.dirname(PARAMS_OUT_PATH.format(dataset_id=dataset_id))
    os.makedirs(out_dir, exist_ok=False)
    # Store preprocessing configuration (flags)
    with open(PARAMS_OUT_PATH.format(dataset_id=dataset_id), "w") as params_f:
        params = {k: v.value for k, v in FLAGS._flags().items()}
        json.dump(params, params_f, indent=2)
    # Process answer pairs
    with open(FLAGS.jsonl_path, "r") as jsonl_file,\
        open(LOCAL_TSV_PATH.format(dataset_id=dataset_id, split="train"), "w") as train_anss_f,\
        open(LOCAL_TSV_PATH.format(dataset_id=dataset_id, split="val"), "w") as val_anss_f,\
        open(LOCAL_TSV_PATH.format(dataset_id=dataset_id, split="test"), "w") as test_anss_f:
        split_to_file = {
            "train": train_anss_f,
            "val": val_anss_f,
            "test": test_anss_f
        }
        for line in tqdm(jsonl_file):
            question = json.loads(line)
            if not question_is_valid(question):
                continue
            for ans1_idx, ans1 in enumerate(question["good_comments"]):
                if not answer_is_valid(ans1):
                    continue
                for ans2 in question["good_comments"][ans1_idx + 1:]:
                    if not answer_is_valid(ans2):
                        continue
                    # Answers and question valid, is answer pair valid?
                    if answer_pair_is_valid(
                        ans1, ans2,
                        max_time_diff=FLAGS.max_time_diff,
                        max_len_ratio=FLAGS.max_len_ratio,
                        min_score_ratio=FLAGS.min_score_ratio
                        ):
                        dataset_line = to_tsv_line(question, ans1, ans2)
                        split_file = split_to_file[question["split"]]
                        split_file.write(dataset_line + "\n")