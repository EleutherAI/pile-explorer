import argparse
import lm_dataformat as lmd
import gensim
import itertools

from collections import defaultdict
from gensim.corpora.dictionary import Dictionary
from gensim.utils import simple_tokenize
from gensim.models import LdaModel

def chunks(iterable, size=10):
    iterator = iter(iterable)
    for first in iterator:
        yield itertools.chain([first], itertools.islice(iterator, size - 1))

parser = argparse.ArgumentParser(description='Train topic models on a component.')

parser.add_argument('--chunk_size', type=int, default=2**12,
                    help='Number of documents in a chunk')
parser.add_argument('--threshold', type=int, default=2**10, help='Number of documents to run another online batch')
parser.add_argument('--input_path', required=True, help='Path to Pile split to use for dictionary (validation, ideally)')
parser.add_argument('--dictionary_file', required=True, help='File name to load the dictionary from')
parser.add_argument('--num_topics', type=int, default=32, help='Number of topics to use for LDA')
parser.add_argument('--component', required=True, help='Component tag for the pile component to run the model on')

args = parser.parse_args()

input_path = args.input_path
dictionary_file = args.dictionary_file
COMPONENT = args.component
N_TOPICS = args.num_topics
CHUNK_SIZE = args.chunk_size
THRESHOLD = args.threshold

dictionary = Dictionary.load(dictionary_file)

rdr = lmd.Reader(input_path)
gnr = rdr.stream_data(get_meta=True)
doc_chunks = chunks(gnr, size=CHUNK_SIZE)

component_docs = []

model = None

progress = 0
count = 0

for chunk in doc_chunks:
    print(f'{count:,} documents processed from {COMPONENT} out of {progress:,} total documents' )

    [component_docs.append(dictionary.doc2bow(simple_tokenize(text))) for (text, meta) in chunk if meta['pile_set_name'] == COMPONENT]

    if len(component_docs) > THRESHOLD:
        if model:
            #pass
            model.update(component_docs)
        else:
            #print(key)
            model = LdaModel(component_docs, id2word=dictionary, num_topics=N_TOPICS, iterations=100, chunksize=CHUNK_SIZE / 2)
        count += len(component_docs)
        component_docs = []

    progress = progress + CHUNK_SIZE

model.update(component_docs)
component_docs = []

model.save(COMPONENT + '.model.topic')
