import argparse
import lm_dataformat as lmd
import gensim
import itertools

from gensim.corpora.dictionary import Dictionary
from gensim.utils import simple_tokenize

# Utility to chunk a generator
def chunks(iterable, size=10):
    iterator = iter(iterable)
    for first in iterator:
        yield itertools.chain([first], itertools.islice(iterator, size - 1))

parser = argparse.ArgumentParser(description='Build a dictionary for topic modeling.')

parser.add_argument('--chunk_size', type=int, default=1000,
                    help='Number of documents in a chunk')
parser.add_argument('--input_path', help='Path to Pile split to use for dictionary (validation, ideally)')
parser.add_argument('--output_name', help='File name to give the dictionary upon saving')

args = parser.parse_args()

input_path = args.input_path
output_name = args.output_name
CHUNK_SIZE = args.chunk_size

# Stream in documents from path
rdr = lmd.Reader(input_path)
gnr = rdr.stream_data(get_meta=True)

# Build a dictionary out of the validation documents
dictionary = Dictionary()
docs = rdr.stream_data(threaded=True)
doc_chunks = chunks(docs, size=CHUNK_SIZE)
# Progress in chunks
for chunk in doc_chunks:
    print("Adding ", CHUNK_SIZE, " docs")
    dictionary.add_documents([simple_tokenize(doc) for doc in list(chunk)])

# Keep only 2**16 most frequent tokens
dictionary.filter_extremes(keep_n=2**16)
dictionary.compactify()
dictionary.save(output_name)
