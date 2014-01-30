"""
USAGE: python %(program)s DICTIONARY_FILE VECTORS_FILE NUM_TOPICS NUM_ITERATIONS

Example: python run_lda.py yelp_out_wordids.txt.bz2 yelp_out_tfidf.mm 50 10
"""

import logging, gensim, bz2
import sys

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    if len(sys.argv) < 5:
        print globals()['__doc__'] % locals()
        sys.exit(1)
    else:
        id2word = gensim.corpora.Dictionary.load_from_text(sys.argv[1])
        mm = gensim.corpora.MmCorpus(sys.argv[2])
        lda = gensim.models.ldamodel.LdaModel(corpus=mm, id2word=id2word,
                num_topics=sys.arv[3], update_every=1,
                chunksize=10000, passes=sys.argv[4])
        lda.save('topics_' + sys.argv[3] + '_' + sys.argv[4])
