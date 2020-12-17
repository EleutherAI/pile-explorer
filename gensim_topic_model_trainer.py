import argparse
import lm_dataformat as lmd
import gensim
import itertools

from multiprocessing import Pool
from collections import defaultdict
from gensim.corpora.dictionary import Dictionary
from gensim.utils import simple_tokenize
from gensim.models import LdaModel, LdaMulticore
from slugify import slugify

def chunks(iterable, size=10):
    iterator = iter(iterable)
    for first in iterator:
        yield itertools.chain([first], itertools.islice(iterator, size - 1))

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
parser.add_argument('--dictionary_file', required=True, help='File name to load the dictionary from')
parser.add_argument('--num_topics', type=int, default=32, help='Number of topics to use for LDA')
parser.add_argument('--component', required=True, choices=component_options, help='Component tag for the pile component to run the model on')
parser.add_argument('--chunk_size', type=int, default=2**12,
                    help='Number of documents in a chunk')
parser.add_argument('--threshold', type=int, default=2**10, help='Number of documents to run another online batch')
parser.add_argument('--tokenizer_processes', type=int, default=1, help='Number of processes for input tokenization')
parser.add_argument('--lda_processes', type=int, default=1, help='Number of processes for LDA training')

args = parser.parse_args()

input_path = args.input_path
dictionary_file = args.dictionary_file
COMPONENT = args.component
N_TOPICS = args.num_topics
CHUNK_SIZE = args.chunk_size
THRESHOLD = args.threshold
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

count = 0

with Pool(TOK_PROCESSES) as p:
    doc_iter = p.imap_unordered(baggify, stream, chunksize=CHUNK_SIZE)
    for (i, (bow, component)) in enumerate(doc_iter):
        print(f'Read {i:,} documents so far')
        if bow:
            docs[component].append(bow)

            if len(docs[component]) >= THRESHOLD:
                if component in models:
                    print(f'Training model for {component} component')
                    models[component].update(docs[component])
                else:
                    print(f'Initializing model for {component} component')
                    if LDA_PROCESSES < 2:
                        models[component] = LdaModel(docs[component],
                                                     id2word=dictionary,
                                                     num_topics=N_TOPICS,
                                                     iterations=50)
                    else:
                        models[component] = LdaMulticore(docs[component],
                                                         id2word=dictionary,
                                                         num_topics=N_TOPICS,
                                                         iterations=50,
                                                         workers=LDA_PROCESSES-1)

                count += len(docs[component])
                print(f'Processed {count:,} documents so far')
                docs[component] = []

for component in component_list:
    if component in models:
        models[component].update(docs[component])
    else:
        print(f'Initializing model for {component} component')
        if LDA_PROCESSES > 1:
            models[component] = LdaModel(docs[component],
                                         id2word=dictionary,
                                         num_topics=N_TOPICS,
                                         iterations=50)
        else:
            models[component] = LdaMulticore(docs[component],
                                             id2word=dictionary,
                                             num_topics=N_TOPICS,
                                             iterations=50,
                                             workers=LDA_PROCESSES-1)

    models[component].save(slugify(component) + '.model.topic')
    docs[component] = None
