import os
import re
import json
from datetime import datetime

import platform
import socket
from werkzeug.serving import BaseWSGIServer
import logging
import flask
from flask_cors import CORS

import tensorflow as tf

from t5.models.mtf_model import MtfModel
from reward.comparative.model import ComparativeRewardModel
from best_of_n.generator import BestOfNGenerator

SAMPLING_KEEP_TOP_P = 0.95
BEST_OF_N_N = 128
T5_MODEL_DIR = "gs://seri2021-advice-eu/turingadvice/baselines/t5/11B"
T5_MODEL_CKPT = 1010000
REWARD_MODEL_DIR = "gs://seri2021-advice-eu/turingadvice/reward/comparative/checkpoints/3B/f2-1-small-batch"
REWARD_MODEL_CKPT = 1019348
BoN_TMP_DIR = "gs://seri2021-advice-eu/turingadvice/frontend"
MODEL_PARALLELISM = 8
ITERATIONS_PER_LOOP = 10
TEMPLATE_DIR = "./frontend"

# Initialize models and Best-of-N generator
t5_model = MtfModel(
    model_dir=T5_MODEL_DIR,
    tpu=os.uname()[1],
    tpu_topology="2x2", # Must be this for validation (Rowan)
    model_parallelism=MODEL_PARALLELISM,
    batch_size=1,
    sequence_length={"inputs": 1280, "targets": 512},
    iterations_per_loop=ITERATIONS_PER_LOOP
)
reward_model = ComparativeRewardModel(
    model_dir=REWARD_MODEL_DIR,
    tpu=os.uname()[1],
    tpu_topology="2x2", # Must be this for validation (Rowan)
    model_parallelism=MODEL_PARALLELISM,
    batch_size=1,
    sequence_length={"inputs": 1280, "targets": 512},
    iterations_per_loop=ITERATIONS_PER_LOOP
)
BoN_generator = BestOfNGenerator(
    t5_model=t5_model,
    t5_model_ckpt_steps=T5_MODEL_CKPT,
    reward_model=reward_model,
    reward_model_ckpt_steps=REWARD_MODEL_CKPT,
    N=BEST_OF_N_N,
    sampling_keep_top_p=SAMPLING_KEEP_TOP_P,
    tmp_dir=BoN_TMP_DIR
)

# Initialize API
app = flask.Flask(__name__, template_folder=TEMPLATE_DIR)
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
platform_name = platform.system()
if platform_name == "Linux" or platform_name == "Windows":
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1),
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 1),
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 3),
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 5),
elif platform_name == "Darwin":
    TCP_KEEPALIVE = 0x10
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    sock.setsockopt(socket.IPPROTO_TCP, TCP_KEEPALIVE, 3)
CORS(app, resources={r'/api/*': {'origins': '*'}})
logger = logging.getLogger(__name__)

def _datetime_to_str(date):
    return [
        'January', 'February', 'March', 'April', 'May', 'June', 'July',
        'August', 'September', 'October', 'November', 'December'
        ][date.month - 1] + ' {}, {}'.format(date.day, date.year)

@app.route('/api/askbatch', methods=['POST'])
def api_askbatch():
    request_dict = dict(flask.request.json)
    instances = request_dict["instances"]
    date = datetime.utcnow()
    date_str = _datetime_to_str(date)
    for instance in instances:
        instance["date"] = date_str
    advices = BoN_generator.generate_from_instances(instances)
    advices = [re.sub(r'\s+»\s+', '\n\n', advice).strip() for advice in advices]
    request_dict.update({"advices": advices})
    with tf.io.gfile.GFile(os.path.join(BoN_TMP_DIR, "log.jsonl"), "a+") as logfile:
        logfile.write(json.dumps(request_dict) + "\n")
    return flask.jsonify({"gens": advices}), 200

if __name__ == "__main__":
    try:
        bind_to = ("0.0.0.0", 5000)
        sock.bind(bind_to)
        sock.listen()
        sock_fd = sock.fileno()
        logging.info("Sock FD is {}".format(sock_fd))
        base_wsgi = BaseWSGIServer(*bind_to, app, fd=sock_fd)
        base_wsgi.serve_forever()
    finally:
        if not sock._closed:
            msg = "Socket not closed, closing."
            sock.close()
        else:
            msg = "Socket already closed."
        logger.info(msg)

