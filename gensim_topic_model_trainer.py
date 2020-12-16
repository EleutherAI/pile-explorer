import argparse
import lm_dataformat as lmd
import gensim
import itertools

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
parser.add_argument('--cores', type=int, default=1, help='Number of CPU cores')

args = parser.parse_args()

input_path = args.input_path
dictionary_file = args.dictionary_file
COMPONENT = args.component
N_TOPICS = args.num_topics
CHUNK_SIZE = args.chunk_size
THRESHOLD = args.threshold
CORES = args.cores

if COMPONENT == 'all':
    components = component_list
else:
    components = [COMPONENT]

dictionary = Dictionary.load(dictionary_file)

rdr = lmd.Reader(input_path)
stream = rdr.stream_data(get_meta=True)
gnr = ((dictionary.doc2bow(simple_tokenize(text)), meta['pile_set_name']) for (text, meta) in stream)
doc_chunks = chunks(gnr, size=CHUNK_SIZE)

docs = defaultdict(list)
models = defaultdict()

progress = 0
count = 0

for chunk in doc_chunks:
    if count >= 100_000:
        break
    print(f'{count:,} documents trained on out of {progress:,} documents read')

    if len(components) == 22:
        [docs[component].append(bow) for (bow, component) in chunk]
    else:
        [docs[component].append(bow) for (bow, component) in chunk if component in components]   

    for component in components:
        if len(docs[component]) > THRESHOLD:
            if component in models.keys():
                print(f'Training model for {component} component')
                models[component].update(docs[component])
            else:
                print(f'Initializing model for {component} component')
                if CORES > 1:
                    models[component] = LdaModel(docs[component],
                                         id2word=dictionary,
                                         num_topics=N_TOPICS,
                                         iterations=100,
                                         chunksize=CHUNK_SIZE / 2)
                else:
                    models[component] = LdaMulticore(docs[component],
                                                     id2word=dictionary,
                                                     num_topics=N_TOPICS,
                                                     iterations=100,
                                                     chunksize=CHUNK_SIZE / 2,
                                                     workers=CORES-1)

            count += len(docs[component])
            docs[component] = []

    progress = progress + CHUNK_SIZE

for component in component_list:
    model[component].update(docs[component])
    model.save(slugify(component) + '.model.topic')
    docs[component] = None
