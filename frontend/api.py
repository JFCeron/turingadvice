import os
import re
import json
import logging
from datetime import datetime

import flask
from flask_cors import CORS
import click
from gevent.pywsgi import WSGIServer

from t5.models.mtf_model import MtfModel
from reward.comparative.model import ComparativeRewardModel
from best_of_n.generator import BestOfNGenerator

SAMPLING_KEEP_TOP_P = 0.95
BEST_OF_N_N = 80
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
    with open(os.path.join(BoN_TMP_DIR, "log.jsonl"), "a+") as logfile:
        logfile.write(json.dumps(request_dict) + "\n")
    return flask.jsonify({"gens": advices}), 200

@click.command()
def serve():
    """Serve predictions on port 5000."""
    logging.basicConfig(
        format='%(asctime)s:%(levelname)s:%(name)s:%(message)s',
        level=logging.INFO)
    logger.info('Running prod server on http://127.0.0.1:5000/')

WSGIServer(('0.0.0.0', 5000), app).serve_forever()
