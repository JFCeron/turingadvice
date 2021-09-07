"""
Use this script to get all the model advice, and the reddit advice
"""
import sys

from tqdm import tqdm
import json
# Enable all logging
import logging
import argparse
import os
import asyncio
import aiohttp
import re
import random
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('-date_tag', type=str, default="Sep-03-21", help='Default date tag to use')

args = parser.parse_args()

handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
logger = logging.getLogger('prawcore')
logger.setLevel(logging.DEBUG)


model_to_url = {
    # 'T5-11B': 'http://34.91.186.193:5000/api/askbatch',
    'Ethan-Juan-Suma': 'http://34.91.27.43:5000/api/askbatch',
}


async def generate_from_model_async(items, model_type: str):
    """
    Generates from all of the models helper function, in parallel
    :param items:
    :param model_type:
    :return:
    """
    data = {
        'instances': [{'title': x['title'], 'selftext': x['selftext'],
                       'subreddit': x['subreddit']} for x in items],
        'target': 'advice',
    }
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(force_close=False), timeout=aiohttp.ClientTimeout(total=12 * 60 * 60)) as session:
        async with session.post(model_to_url[model_type], json=data) as resp:
            response = await resp.text()
    # print(f"RESPONSE FOR {model_type} IS {response}", flush=True)
    response = json.loads(response)
    print(f"DONE WITH {model_type}", flush=True)
    if 'gens' not in response:
        raise ValueError("Malformed response?\n{}".format(response))
    return tuple(response['gens'])


def generate_from_all_models(items):
    """
    Generates from all of the models, in parallel
    :param items: List of {'title': str, 'selftext': str, 'subreddit': str} things
    :return:
    """
    loop = asyncio.get_event_loop()
    model_list = sorted(model_to_url.keys())
    task_list = [generate_from_model_async(items, model_type=m) for m in model_list]
    result_list = loop.run_until_complete(asyncio.gather(*task_list))
    loop.close()
    return {k: r for k, r in zip(model_list, result_list)}


cache_fn = f'200questions.jsonl'
if os.path.exists(cache_fn):
    print(f"Cache fn {cache_fn} exists")
    info = []
    with open(cache_fn, 'r') as f:
        for l in f:
            info.append(json.loads(l))

results = generate_from_all_models(info)
for i, x in enumerate(info):
    x['modeladvice'] = {model_name: model_res[i] for model_name, model_res in results.items()}

print(results)