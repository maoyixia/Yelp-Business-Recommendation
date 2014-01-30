from mrjob.job import MRJob
from mrjob.protocol import JSONValueProtocol
import sys

sys.stdout = open('reviews/categorized_reviews.json', 'w')

class CreateCategoryReview(MRJob):
  INPUT_PROTOCOL = JSONValueProtocol

  def review_category_mapper(self, _, data):
    if data['type'] == 'review':
      yield data['business_id'], ('review', (data['text'], data['stars']))
    elif data['type'] == 'business':
      if data['categories']:
        yield data['business_id'], ('categories', data['categories']) 

  def category_join_reducer(self, business_id, reviews_or_categories):
    categories = None
    reviews = []
    for data_type, data in reviews_or_categories:
      if data_type == 'review':
        reviews.append(data)
      else:
        categories = data
  
    if not categories:
      return

    for category in categories:
      for review in reviews:
        yield None, {'category': category, 'business_id': business_id,
            'text': review[0], 'stars': review[1]}

  def steps(self):
    return [ self.mr(self.review_category_mapper, self.category_join_reducer) ]

if __name__ == '__main__':
  CreateCategoryReview().run()
