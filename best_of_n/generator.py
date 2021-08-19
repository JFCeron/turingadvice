import os

import tensorflow as tf

class BestOfNGenerator():
    def __init__(
        self, N, t5_model, t5_model_ckpt_steps, sampling_keep_top_p,
        reward_model, reward_model_ckpt_steps, tmp_dir
        ):
        self.N = N
        self.t5_model = t5_model
        self.t5_model_ckpt_steps = t5_model_ckpt_steps
        self.sampling_keep_top_p = sampling_keep_top_p
        self.reward_model = reward_model,
        self.reward_model_ckpt_steps = reward_model_ckpt_steps,
        self.tmp_dir = tmp_dir
    
    def generate_N(self, inputs_path, outputs_path):
        # Repeat each input N times, store in temporary file
        TMP_REPEATS_PATH = os.path.join(self.tmp_dir, "BoN-repeats.txt")
        with tf.io.gfile.GFile(inputs_path, "r") as inputs_file, \
             tf.io.gfile.GFile(TMP_REPEATS_PATH, "w") as repeats_file:
            for line in inputs_file:
                for _ in range(self.N):
                    repeats_file.write(line)
        # Predict over repeated inputs file
        self.t5_model.predict(
            input_file=TMP_REPEATS_PATH,
            output_file=outputs_path,
            checkpoint_steps=self.t5_model_ckpt_steps,
            sampling_keep_top_p=self.sampling_keep_top_p
        )
    
    def generate(self, inputs_path, outputs_path):
        TMP_N_GENERATIONS_PATH = os.path.join(self.tmp_dir, "N-generations.txt")
        TMP_N_SCORES_PATH = os.path.join(self.tmp_dir, "N-scores.txt")
        self.generate_N(inputs_path, TMP_N_GENERATIONS_PATH)
        self.reward_model.predict_from_file(
            input_path=TMP_N_GENERATIONS_PATH,
            output_path=TMP_N_SCORES_PATH,
            checkpoint_steps=self.reward_model_ckpt_steps
        )
        # Write top-scoring answer to every question
        with tf.io.gfile.GFile(TMP_N_GENERATIONS_PATH, "r") as gens_file, \
             tf.io.gfile.GFile(TMP_N_SCORES_PATH, "r") as scores_file, \
             tf.io.gfile.GFile(outputs_path, "w") as outputs_file:
            best_score = -float("inf")
            best_answer = None
            answer_count = 0
            for answer, str_score in zip(gens_file, scores_file):
                score = float(str_score)
                answer_count += 1
                if score > best_score:
                    best_score = score
                    best_answer = answer
                if answer_count >= self.N:
                    # Write best answer and start processing next question
                    outputs_file.write(best_answer)
                    answer_count = 0
                    best_score = -float("inf")
                    best_answer = None