import os
import sys
from absl import flags

import tensorflow as tf

from t5.models.mtf_model import MtfModel, _get_latest_checkpoint_from_dir
from best_of_n.generator import BestOfNGenerator

def _define_flags():
    flags.DEFINE_string(
        name="input_path",
        default=None,
        help="Path to a tab-separated text file with columns [inputs, targets]"
    )
    flags.DEFINE_string(
        name="output_path",
        default=None,
        help="File to store predictions, one per line of input"
    )
    flags.DEFINE_integer(
        name="N",
        default=1,
        help="How many outputs to generate per input?"
    )
    flags.DEFINE_string(
        name="model_dir",
        default=None,
        help="Directory with T5 checkpoints"
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
    flags.DEFINE_integer(
        name="checkpoint_steps",
        default=-1,
        help="Steps in checkpoint to be used for prediction"
    )
    flags.DEFINE_string(
        name="tmp_dir",
        default=None,
        help="Temporary dir for internal use of BestOfNGenerator"
    )

def main(_):
    FLAGS = _define_flags()
    FLAGS(sys.argv)
    if FLAGS.checkpoint_steps == -1:
        ckpt_steps = _get_latest_checkpoint_from_dir(FLAGS.model_dir)
    else:
        ckpt_steps = FLAGS.checkpoint_steps
    t5_model = MtfModel(
        model_dir=FLAGS.model_dir,
        tpu=os.uname()[1],
        tpu_topology='2x2', # Must be this for validation
        model_parallelism=FLAGS.model_parallelism,
        batch_size=FLAGS.batch_size,
        sequence_length={"inputs": 1280, "targets": 512},
        iterations_per_loop=FLAGS.iterations_per_loop,
    )
    generator = BestOfNGenerator(
        t5_model=t5_model,
        t5_model_ckpt_steps=ckpt_steps,
        N=FLAGS.N,
        sampling_keep_top_p=0.94,
        tmp_dir=FLAGS.tmp_dir
    )
    generator.generate_N(FLAGS.input_path, FLAGS.output_path)

if __name__ == "__main__":
    tf.app.run()