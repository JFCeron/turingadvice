import tensorflow as tf

from reward.comparative.data.tsvs_to_tfrecords import TFRECORDS_PATH, SEQUENCE_LENGTH

def get_dataset(split, batch_size):
    tfrecords_path = TFRECORDS_PATH.format(split=split)
    serialized_dataset = tf.data.TFRecordDataset(tfrecords_path)
    feature_description = {}
    feature_description = {
        "inputs": tf.io.FixedLenFeature(
            SEQUENCE_LENGTH["inputs"],
            tf.int64,
            default_value=[0]*SEQUENCE_LENGTH["inputs"]
        ),
        "inputs_position": tf.io.FixedLenFeature(
            SEQUENCE_LENGTH["inputs"],
            tf.int64,
            default_value=[0]*SEQUENCE_LENGTH["inputs"]
        ),
        "targets1": tf.io.FixedLenFeature(
            SEQUENCE_LENGTH["targets"],
            tf.int64,
            default_value=[0]*SEQUENCE_LENGTH["targets"]
        ),
        "targets1_position": tf.io.FixedLenFeature(
            SEQUENCE_LENGTH["targets"],
            tf.int64,
            default_value=[0]*SEQUENCE_LENGTH["targets"]
        ),
        "targets2": tf.io.FixedLenFeature(
            SEQUENCE_LENGTH["targets"],
            tf.int64,
            default_value=[0]*SEQUENCE_LENGTH["targets"]
        ),
        "targets2_position": tf.io.FixedLenFeature(
            SEQUENCE_LENGTH["targets"],
            tf.int64,
            default_value=[0]*SEQUENCE_LENGTH["targets"]
        ),
        "inputs_segmentation": tf.io.FixedLenFeature(
            SEQUENCE_LENGTH["inputs"],
            tf.int64,
            default_value=[0]*SEQUENCE_LENGTH["inputs"]
        ),
        "targets1_segmentation": tf.io.FixedLenFeature(
            SEQUENCE_LENGTH["targets"],
            tf.int64,
            default_value=[0]*SEQUENCE_LENGTH["targets"]
        ),
        "targets2_segmentation": tf.io.FixedLenFeature(
            SEQUENCE_LENGTH["targets"],
            tf.int64,
            default_value=[0]*SEQUENCE_LENGTH["targets"]
        )
    }
    tokens_dataset = serialized_dataset.map(
        lambda x: tf.io.parse_single_example(x, feature_description)
    )
    stacked_dataset = tokens_dataset.map(_stack_answer_pairs)
    return stacked_dataset.batch(batch_size)

def _stack_answer_pairs(sample):
    targets = tf.stack([sample["targets1"], sample["targets2"]])
    targets_position = tf.stack([sample["targets1_position"], sample["targets2_position"]])
    targets_segmentation = tf.stack([sample["targets1_segmentation"], sample["targets2_segmentation"]])
    stacked_sample = {
        "inputs": sample["inputs"],
        "inputs_position": sample["inputs_position"],
        "inputs_segmentation": sample["inputs_segmentation"],
        "targets": targets,
        "targets_position": targets_position,
        "targets_segmentation": targets_segmentation
    }
    return stacked_sample