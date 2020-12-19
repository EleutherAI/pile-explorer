import argparse
import lm_dataformat as lmd
import gensim
import itertools
import json
import pandas as pd
import seaborn as sns
import numpy as np

from math import pow
from multiprocessing import Pool
from collections import defaultdict
from gensim.corpora.dictionary import Dictionary
from gensim.utils import simple_tokenize
from gensim.models import LdaModel, LdaMulticore

component_list = {
    'StackExchange',
    'PubMed Central',
    'HackerNews',
    'OpenWebText2',
    'Enron Emails',
    'Ubuntu IRC',
    'BookCorpus',
    'USPTO',
    'Gutenberg (PG-19)',
    'Github',
    'CommonCrawl',
    'PubMed Abstracts',
    'FreeLaw',
    'Wikipedia (en)',
    'ArXiv',
    'DM Mathematics',
    'PhilPapers',
    'EuroParl',
    'Bibliotik',
    'OpenSubtitles',
    'YoutubeSubtitles',
    'NIH ExPorter',
}

component_options = component_list | { 'all' }

parser = argparse.ArgumentParser(description='Train topic models on a component.')

parser.add_argument('--input_path', required=True, help='Path to Pile split to use for dictionary (validation, ideally)')
parser.add_argument('--models_path', required=True, help='Path to saved LDA models')
parser.add_argument('--dictionary_file', required=True, help='File name to load the dictionary from')
parser.add_argument('--component', required=True, choices=component_options, help='Component tag for the pile component to run the model on')
parser.add_argument('--chunk_size', type=int, default=2**12,
                    help='Number of documents in a chunk')
parser.add_argument('--tokenizer_processes', type=int, default=1, help='Number of processes for input tokenization')
parser.add_argument('--lda_processes', type=int, default=1, help='Number of processes for LDA training')

args = parser.parse_args()

input_path = args.input_path
models_path = args.models_path
dictionary_file = args.dictionary_file
COMPONENT = args.component
CHUNK_SIZE = args.chunk_size
TOK_PROCESSES = args.tokenizer_processes
LDA_PROCESSES = args.lda_processes

if COMPONENT == 'all':
    components = component_list
else:
    components = [COMPONENT]

dictionary = Dictionary.load(dictionary_file)

rdr = lmd.Reader(input_path)
stream = rdr.stream_data(get_meta=True)

def baggify(item):
    text, meta = item
    component = meta['pile_set_name']
    if component in components:
        bow_or_none = dictionary.doc2bow(simple_tokenize(text))
    else:
        bow_or_none = None
    return (bow_or_none, component)

docs = defaultdict(list)
models = defaultdict()

print("Loading models...")
for component in components:
    models[component] = LdaMulticore.load(models_path + component + '.model.topic')

with Pool(TOK_PROCESSES) as p:
    print("Reading documents...")
    doc_iter = p.imap_unordered(baggify, stream, chunksize=CHUNK_SIZE)
    for (i, (bow, component)) in enumerate(doc_iter):
        #print(f'Read {i:,} documents so far')
        if bow:
            docs[component].append(bow)        

perplexities = defaultdict(dict)

for (training_corpus, eval_corpus) in itertools.product(*(components, components)):
    print(f'Computing perlexity for {training_corpus} on {eval_corpus}...')
    perplexity = pow(2, -models[training_corpus].log_perplexity(docs[eval_corpus]))
    perplexities[training_corpus][eval_corpus] = perplexity
    
with open('topic_model_perplexities.json', 'w') as fp:
    json.dump(perplexities, fp)
    print("Saving topic modeling cross-corpus perplexities...")
    
df = pd.DataFrame.from_dict(perplexities,
                   orient='index')
log_perplexities = df.apply(np.log)

ax = sns.heatmap(log_perplexities, xticklabels=True, yticklabels=True)
ax.set_xlabel("Testing Corpus")
ax.set_ylabel("Training Corpus")

fig = ax.get_figure()
fig.savefig('topic_model_perplexities.png')