import json
import logging
import multiprocessing
import re

from gensim import utils

from gensim.corpora.dictionary import Dictionary
from gensim.corpora.textcorpus import TextCorpus

from nltk import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

logger = logging.getLogger('gensim.corpora.wikicorpus')

stwords = stopwords.words('english')
norm = lambda word: re.sub('[^a-z]', '', word.lower())
stemmer = PorterStemmer()

class YelpCorpus(TextCorpus):

    def __init__(self, fname, processes=None, lemmatize=utils.HAS_PATTERN, dictionary=None):
        """
        Initialize the corpus. Unless a dictionary is provided, this scans the
        corpus once, to determine its vocabulary.

        If `pattern` package is installed, use fancier shallow parsing to get
        token lemmas. Otherwise, use simple regexp tokenization. You can override
        this automatic logic by forcing the `lemmatize` parameter explicitly.

        """
        self.fname = fname
        if processes is None:
            processes = max(1, multiprocessing.cpu_count() - 1)
        self.processes = processes
        self.lemmatize = lemmatize
        if dictionary is None:
            self.dictionary = Dictionary(self.get_texts())
        else:
            self.dictionary = dictionary

    def get_texts(self):
        """
        Iterate over the dump, returning text version of each article as a list
        of tokens.
        """

        reviews = 0
        positions = 0
        texts = [text for text in _extract_reviews(self.fname)]
        pool = multiprocessing.Pool(self.processes)
        # process the corpus in smaller chunks of docs, because multiprocessing.Pool
        # is dumb and would load the entire input into RAM at once...
        #for group in utils.chunkize(texts, chunksize=10 * self.processes, maxsize=1):
        for group in utils.chunkize(texts, chunksize=10 * self.processes, maxsize=1):
            for tokens in pool.imap(process_review, group): # chunksize=10):
                reviews += 1
                positions += len(tokens)
                yield tokens
        pool.terminate()

        logger.info("finished iterating over the generated Yelp corpus of %i documents with %i positions"
                " (total %i articles, %i positions before pruning articles shorter than %i words)" %
                (reviews, positions, reviews, positions, 10000))
        self.length = reviews # cache corpus length

def process_review(review):
    #return [token.encode('utf8') for token in utils.tokenize(review, lower=True, errors='ignore')
    #        if 2 <= len(token) <= 15]
    tokens = [token.encode('utf8') for token in utils.tokenize(review, lower=True, errors='ignore')
            if 2 <= len(token) <= 15]
    tokens = [norm(token) for token in tokens if norm(token)]
    tokens = [token for token in tokens if token not in stwords]
    tokens = [stemmer.stem(token) for token in tokens if stemmer.stem(token)]
    return tokens

def _extract_reviews(fname):
    with open(fname, 'r') as reviews:
        for review in reviews:
            data = json.loads(review)
            yield data['text']
