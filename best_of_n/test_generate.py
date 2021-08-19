import os
import sys
from absl import flags

import tensorflow as tf

from t5.models.mtf_model import MtfModel, _get_latest_checkpoint_from_dir
from reward.comparative.model import ComparativeRewardModel
from best_of_n.generator import BestOfNGenerator

def _define_flags():
    flags.DEFINE_string(
        name="input_path",
        default=None,
        help="Path to a tab-separated text file with columns [inputs]"
    )
    flags.DEFINE_string(
        name="output_path",
        default=None,
        help="File to store predictions, one per line of input"
    )
    flags.DEFINE_integer(
        name="N",
        default=1,
        help="N for best-of-N algorithm"
    )
    flags.DEFINE_string(
        name="t5_model_dir",
        default=None,
        help="Directory with T5 checkpoints"
    )
    flags.DEFINE_integer(
        name="t5_checkpoint_steps",
        default=-1,
        help="Steps in checkpoint to be used for t5 generation"
    )
    flags.DEFINE_string(
        name="reward_model_dir",
        default=None,
        help="Directory with T5 checkpoints"
    )
    flags.DEFINE_integer(
        name="reward_checkpoint_steps",
        default=-1,
        help="Steps in checkpoint to be used for reward scoring"
    )
    flags.DEFINE_integer(
        name="iterations_per_loop",
        default=1000,
        help="How many steps to make in each estimator call."
    )
    flags.DEFINE_integer(
        name="model_parallelism",
        default=8,
        help="Number of cores per model instance."
    )
    flags.DEFINE_integer(
        name="batch_size",
        default=1,
        help="Batch size. Spillover samples are ignored"
    )
    flags.DEFINE_string(
        name="tmp_dir",
        default=None,
        help="Temporary dir for internal use of BestOfNGenerator"
    )
    return flags.FLAGS

def main(_):
    FLAGS = _define_flags()
    FLAGS(sys.argv)
    # Initialize T5 model
    if FLAGS.t5_checkpoint_steps == -1:
        t5_ckpt_steps = _get_latest_checkpoint_from_dir(FLAGS.t5_model_dir)
    else:
        t5_ckpt_steps = FLAGS.t5_checkpoint_steps
    t5_model = MtfModel(
        model_dir=FLAGS.t5_model_dir,
        tpu=os.uname()[1],
        tpu_topology='2x2', # Must be this for validation
        model_parallelism=FLAGS.model_parallelism,
        batch_size=FLAGS.batch_size,
        sequence_length={"inputs": 1280, "targets": 512},
        iterations_per_loop=FLAGS.iterations_per_loop,
    )
    # Initialize reward model
    if FLAGS.t5_checkpoint_steps == -1:
        reward_ckpt_steps = _get_latest_checkpoint_from_dir(FLAGS.reward_model_dir)
    else:
        reward_ckpt_steps = FLAGS.reward_checkpoint_steps
    reward_model = ComparativeRewardModel(
        model_dir=FLAGS.reward_model_dir,
        tpu=os.uname()[1],
        tpu_topology='2x2', # Must be this for validation
        model_parallelism=FLAGS.model_parallelism,
        batch_size=FLAGS.batch_size,
        sequence_length={"inputs": 1280, "targets": 512},
        iterations_per_loop=FLAGS.iterations_per_loop
    )
    # Generate answers
    generator = BestOfNGenerator(
        t5_model=t5_model,
        t5_model_ckpt_steps=t5_ckpt_steps,
        reward_model=reward_model,
        reward_model_ckpt_steps=reward_ckpt_steps,
        N=FLAGS.N,
        sampling_keep_top_p=0.94,
        tmp_dir=FLAGS.tmp_dir
    )
    generator.generate(FLAGS.input_path, FLAGS.output_path)

if __name__ == "__main__":
    tf.app.run()