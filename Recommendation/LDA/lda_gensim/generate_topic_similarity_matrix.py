import logging, gensim, bz2

if __name__ == '__main__':
  lda = gensim.models.ldamodel.LdaModel.load('lda_topic50_chunk500_pass100')
  topics = lda.show_topics(-1, 100)
  formatted_topics = []
  for topic in topics:
    formatted_topic = {}
    pairs = topic.split('+')
    for pair in pairs:
      weight, word = pair.split('*')
      formatted_topic[word] = float(weight)
    formatted_topics.append(formatted_topic)
  
  similarity_matrix = []
  for topic1 in formatted_topics:
    row = []
    words1 = set(topic1.keys())
    for topic2 in formatted_topics:
      words2 = set(topic2.keys())
      common_words = words1.intersection(words2)
      dot_product = 0.0
      for word in common_words: 
        dot_product += topic1[word] * topic2[word]
      topic1_length = 0.0
      topic2_length = 0.0
      for word in topic1.keys():
        topic1_length += topic1[word] * topic1[word]
      topic1_length = pow(topic1_length, 0.5)
      for word in topic2.keys():
        topic2_length += topic2[word] * topic2[word]
      topic2_length = pow(topic2_length, 0.5)
      row.append(dot_product / (topic1_length * topic2_length))
    similarity_matrix.append(row)
  with open('similarity_matrix.js', 'w') as output:
    output.write('var similarity_matrix = ')
    output.write(str(similarity_matrix))
    output.write(';')
    output.flush()
