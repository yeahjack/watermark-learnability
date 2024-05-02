import argparse
import os

import json

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from watermarks.kth.detect import detect


device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, required=True)
parser.add_argument("--tokenizer_name", type=str, default=None)
parser.add_argument("--dataset_config_name", type=str, default=None)
parser.add_argument("--dataset_split", type=str, default="test")
parser.add_argument("--dataset_num_skip", type=int, default=0)
parser.add_argument("--data_field", type=str, default="text")
parser.add_argument("--num_samples", type=int, default=10000)
parser.add_argument("--prompt_length", type=int, default=50)
parser.add_argument("--completion_length", type=int, default=200)
parser.add_argument("--key_len", type=int, default=256)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--streaming", action="store_true", default=False)
parser.add_argument("--output_file", type=str, required=True)
parser.add_argument("--overwrite_output_file", action="store_true", default=False)
parser.add_argument("--gamma", type=float, default=0.0)

args = parser.parse_args()

if os.path.exists(args.output_file) and not args.overwrite_output_file:
    raise ValueError(f"Output file {args.output_file} already exists and overwrite_output_file is False")

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
vocab_size = len(tokenizer)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset(args.dataset_name, args.dataset_config_name, split=args.dataset_split, streaming=args.streaming)

max_length = args.prompt_length + args.completion_length
min_length = args.completion_length

def filter_length(example):
    return len(tokenizer(example[args.data_field], truncation=True, max_length=max_length)["input_ids"]) >= min_length

if args.dataset_num_skip > 0:
    dataset = dataset.skip(args.dataset_num_skip)

texts = []
for d in dataset:
    if len(texts) >= args.num_samples:
        break
    if filter_length(d):
        texts.append(d[args.data_field])

test_stats = []

rng = np.random.default_rng(args.seed)

def process_text(text, tokenizer, max_length, args, rng, vocab_size):
    tokens = tokenizer.encode(text, return_tensors='np', truncation=True, max_length=max_length)[0]
    random_xi = rng.random((args.key_len, vocab_size)).astype(np.float32)
    null_result = detect(tokens[-args.completion_length:], len(random_xi), args.completion_length, random_xi, gamma=args.gamma)
    return null_result

def init_worker(worker_seed):
    np.random.seed(worker_seed)

def generate_test_stats(texts, tokenizer, max_length, args, seed, vocab_size, num_processes):
    rng = np.random.default_rng(seed)
    worker_func = partial(process_text, tokenizer=tokenizer, max_length=max_length, args=args, rng=rng, vocab_size=vocab_size)

    pool = multiprocessing.Pool(processes=num_processes, initializer=init_worker, initargs=(seed,))

    test_stats = []
    with tqdm(total=len(texts), desc="Processing texts", unit="text", ncols=100) as pbar:
        for result in pool.imap(worker_func, texts):
            test_stats.append(result)
            pbar.update()

    pool.close()
    pool.join()

    return test_stats

num_processes = multiprocessing.cpu_count()
test_stats = generate_test_stats(texts, tokenizer, max_length, args, args.seed, vocab_size, num_processes)

# for text in tqdm(texts):
#     tokens = tokenizer.encode(text, return_tensors='np', truncation=True, max_length=max_length)[0]
#     random_xi = rng.random((args.key_len, vocab_size)).astype(np.float32)
#     null_result = detect(tokens[-args.completion_length:], len(random_xi), args.completion_length, random_xi, gamma=args.gamma)
#     test_stats.append(null_result)


output_dict = {
    "test_stat_ref_dist": test_stats,
}
output_dict.update(vars(args))

# os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

with open(args.output_file, "w") as f:
    print(f"Writing output to {args.output_file}")
    json.dump(output_dict, f, indent=4)
