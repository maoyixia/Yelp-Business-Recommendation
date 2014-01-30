import json
import gensim

lda = gensim.models.lsimodel.LsiModel.load('lda_topic50_chunk500_pass100')

def calc_topic_doc_count():
  #lda = gensim.models.lsimodel.LsiModel.load('lda_stopwords_removal_topic50_chunk500_pass100_alpha5')
  topic_doc_count = {}
  mm = gensim.corpora.MmCorpus('yelp_out_tfidf.mm')
  docs = lda[mm]
  for doc in docs:
      for topic, prob in doc:
          if topic in topic_doc_count:
              topic_doc_count[topic] += 1
          else:
              topic_doc_count[topic] = 1
  return topic_doc_count

def generate_json(topic_doc_count):
  topics = []
  with open('topics_to_use.txt', 'r') as input:
    for topic in input:
      topics.append(topic)
    
  topic_json = {}
  topic_json['name'] = 'topics'
  topic_json['children'] = []
  for t in topic_doc_count: 
      d = {}
      d['name'] = topics[t]
      children = []
      topic_dict = {}
      topic_dict['name'] = topics[t]
      topic_dict['size'] = topic_doc_count[t]
      children.append(topic_dict)
      d['children'] = children
      topic_json['children'].append(d)
  with open('data_topic.json', 'w') as outfile:
      json.dump(topic_json, outfile)

  topic_word_distribution = lda.show_topics(-1, 10)
  topic_word_json = {}
  topic_word_json['name'] = 'topics'
  topic_word_json['children'] = []
  for i in range(len(topic_word_distribution)):
      d = {}
      d['name'] = topics[i]
      children = []
      prob_word_strings = topic_word_distribution[i].split('+')
      for prob_word in prob_word_strings:
          pair = prob_word.split('*')
          prob_word_dict = {}
          prob_word_dict['name'] = pair[1]
          prob_word_dict['size'] = int(float(pair[0]) * 100000)
          children.append(prob_word_dict)
      d['children'] = children
      topic_word_json['children'].append(d)
  with open('data_topic_word.json', 'w') as outfile:
      json.dump(topic_word_json, outfile)

if __name__ == '__main__':
  topic_doc_count = calc_topic_doc_count()
  generate_json(topic_doc_count)
