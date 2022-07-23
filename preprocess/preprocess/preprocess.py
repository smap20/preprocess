"""Processing data for pretraining."""

import argparse
import json
import multiprocessing
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
import time

import torch
try:
    import nltk
    nltk_available = True
except ImportError:
    nltk_available = False

from transformers import GPT2Tokenizer
from data import indexed_dataset

class Encoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        Encoder.tokenizer = GPT2Tokenizer.from_pretrained('gpt2-base')

    def encode(self, line):
        data = line.strip()
        doc_ids = Encoder.tokenizer.encode(data)
        doc_ids.append(Encoder.tokenizer.eod)
        
        # split by 128
        doc_ids_split = []
        for i in range(0, len(doc_ids), 128):
            doc_ids_split.append(doc_ids[i:i+128])

        return doc_ids_split, len(line)

def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str, required=True,
                       help='Path to input JSON')
    group.add_argument('--json-keys', nargs='+', default=['text'],
                       help='space separate listed of keys to extract from json')

    group = parser.add_argument_group(title='output data')
    group.add_argument('--output-prefix', type=str, required=True,
                       help='Path to binary output file without suffix')
    group.add_argument('--dataset-impl', type=str, default='mmap',
                       choices=['lazy', 'cached', 'mmap'])

    group = parser.add_argument_group(title='runtime')
    group.add_argument('--workers', type=int, default=1,
                       help='Number of worker processes to launch')
    group.add_argument('--log-interval', type=int, default=10000,
                       help='Interval between progress updates')
    args = parser.parse_args()
    args.keep_empty = False

    args.rank = 0
    args.make_vocab_size_divisible_by = 128

    return args

def main():
    args = get_args()
    startup_start = time.time()

    print("Opening", args.input)
    fin = open(args.input, 'r', encoding='utf-8')

    encoder = Encoder(args)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-base')

    pool = multiprocessing.Pool(args.workers, initializer=encoder.initializer)

    encoded_docs = pool.imap(encoder.encode, fin, 25)

    print(f"Output prefix: {args.output_prefix}")

    output_bin_file = "{}.bin".format(args.output_prefix)
    output_idx_file = "{}.idx".format(args.output_prefix)
    builder = indexed_dataset.make_builder(output_bin_file,
                                            impl=args.dataset_impl,
                                            vocab_size=tokenizer.vocab_size)

    startup_end = time.time()
    proc_start = time.time()
    total_bytes_processed = 0
    print("Time to startup:", startup_end - startup_start)

    for i, (doc, bytes_processed) in enumerate(encoded_docs, start=1):
        total_bytes_processed += bytes_processed
        for sentence in doc:
            builder.add_item(torch.IntTensor(sentence))

        if i % args.log_interval == 0:
            current = time.time()
            elapsed = current - proc_start
            mbs = total_bytes_processed/elapsed/1024/1024
            print(f"Processed {i} documents",
                  f"({i/elapsed} docs/s, {mbs} MB/s).",
                  file=sys.stderr)

    builder.finalize(output_idx_file)

if __name__ == '__main__':
    main()