import json

class ReviewCategorizer(object):
  def __init__(self, reviewFile):
    self.reviewFile = reviewFile

  def categorizeReview(self):
    category = ''
    output = None
    counter = 1
    with open(self.reviewFile, 'r') as reviews:
      for review in reviews:
        data = json.loads(review)
        if data['category'] != category:
          if output:
            output.flush()
            output.close()
          category = data['category']
          output = open('reviews/category' + str(counter) + '.json', 'w')
          counter += 1
        output.write(review)
      output.flush()
      output.close()
      reviews.close()

if __name__ == '__main__':
  ReviewCategorizer('reviews/sorted_category_reviews.json').categorizeReview()
