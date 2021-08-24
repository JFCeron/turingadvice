import sys

from absl import flags
import tensorflow as tf

def _define_flags():
    flags.DEFINE_list(
        name="input_paths",
        default=None,
        help="Comma-separated list of text files (no quotes)"
    )
    flags.DEFINE_list(
        name="n",
        default=None,
        help=" Comma separated list of integers. For each input path, the number of generations per question in that file"
    )
    flags.DEFINE_string(
        name="output_path",
        default=None,
        help="File to store merged generations"
    )
    flags.DEFINE_integer(
        name="N",
        default=1,
        help="The number of generations per question in the output file"
    )
    return flags.FLAGS

if __name__ == "__main__":
    """
    Merge the text generations in several input files into a single output file
    with N generations per question
    """
    FLAGS = _define_flags()
    FLAGS(sys.argv)
    n = [int(i) for i in FLAGS.n]
    assert sum(n) >= FLAGS.N, "There are < N generations among the input files"
    input_files = [tf.io.gfile.GFile(p, "r") for p in FLAGS.input_paths]
    with tf.io.gfile.GFile(FLAGS.output_path, "w") as output_file:
        to_next_question = True
        while(to_next_question):
            # Read all generations for the next question from each input file
            all_answers_to_question = []
            for idx, input_file in enumerate(input_files):
                anss_from_file = [input_file.readline() for _ in range(n[idx])]
                anss_from_file = [a for a in anss_from_file if len(a) > 0]
                all_answers_to_question.append(anss_from_file)
            # Write N answers to output
            anss_to_write = []
            n_anss_found = 0
            for anss_from_file in all_answers_to_question:
                if len(anss_from_file) == 0:
                    continue
                n_to_write = min(FLAGS.N - n_anss_found, len(anss_from_file))
                anss_to_write.extend(anss_from_file[:n_to_write])
                n_anss_found += n_to_write
                if n_anss_found >= FLAGS.N:
                    break
            if n_anss_found == FLAGS.N:
                for ans in anss_to_write:
                    output_file.write(ans)
            else:
                to_next_question = False
    for input_file in input_files:
        input_file.close()