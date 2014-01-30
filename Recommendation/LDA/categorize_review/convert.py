import json
import sys

if __name__ == '__main__':
  if len(sys.argv) != 2:
    print 'Usage: python convert.py filename'
  else:
    with open(sys.argv[1], 'r') as reviews:
      i = 1
      for review in reviews:
        data = json.loads(review)
        output = open('docs/doc' + str(i) + '.txt', 'w') 
        i += 1
        output.write(data['text'].encode('utf-8'))
        output.flush()
        output.close()
