import sys

from absl import flags
import tensorflow as tf

def _define_flags():
    flags.DEFINE_string(
        name="input_path",
        default=None,
        help="Path to a text file"
    )
    flags.DEFINE_string(
        name="output_path",
        default=None,
        help="File to store sampled generations"
    )
    flags.DEFINE_integer(
        name="N",
        default=1,
        help="The number of generations per question in the input file"
    )
    flags.DEFINE_integer(
        name="n",
        default=None,
        help="The number of generations per question in the output file"
    )
    return flags.FLAGS

if __name__ == "__main__":
    """
    Sample n <= N text generations per question from an input file with N
    generations per question
    """
    FLAGS = _define_flags()
    FLAGS(sys.argv)
    with tf.io.gfile.GFile(FLAGS.input_path, "r") as input_file, \
         tf.io.gfile.GFile(FLAGS.output_path, "w") as output_file:
        gens_written = 0
        gens_seen = 0
        for input_line in input_file:
            if gens_written < FLAGS.n:
                output_file.write(input_line)
                gens_written += 1
            gens_seen += 1
            if gens_seen >= FLAGS.N:
                gens_seen = 0
                gens_written = 0

            
