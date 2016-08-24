# get the keyword
import sys
keyword = sys.argv[1]

# be authorized by using key, token
from twython import Twython
APP_KEY='{Consumer Key}'
APP_SECRET='{Consumer Secret}'
OAUTH_TOKEN='{Access Token}'
OAUTH_TOKEN_SECRET='{Access Token Secret}'

# search the keyword in recent and popular tweets
twitter = Twython(APP_KEY, APP_SECRET, OAUTH_TOKEN, OAUTH_TOKEN_SECRET)
tweets = twitter.search(q=keyword, count=100, result_type='mixed')

# count the text's words
from collections import Counter
user_pref_words_list = {}
count = 0
for u_num in range(0, len(tweets['statuses'])):
   text = tweets['statuses'][u_num]['text']
   user_pref_words_list[u_num] = dict(Counter(text.replace('\n', ' ').split(' ')))

# calculate user's preference based on frequency of the number of words by using crab
from scikits.crab.models import MatrixPreferenceDataModel
model = MatrixPreferenceDataModel(user_pref_words_list)

from scikits.crab.metrics import pearson_correlation
from scikits.crab.similarities import UserSimilarity
similarity = UserSimilarity(model, pearson_correlation)

from scikits.crab.recommenders.knn import UserBasedRecommender
recommender = UserBasedRecommender(model, similarity, with_preference=True)

# calculate each word preference average
from collections import defaultdict
words_pref = defaultdict(float)
for u_num, pref_words_list in user_pref_words_list.items():
    for word, count in pref_words_list.items():
       if(words_pref[word]):
          words_pref[word] = (words_pref[word] + float(count)) / 2.0
       else:
          words_pref[word] = float(count)

# sort the words based on preference
import operator
words_pref_sorted = sorted(words_pref.items(), key=operator.itemgetter(1), reverse=True)

words_list = [word_pref[0] for word_pref in words_pref_sorted[:20]]
words_pref_list = [word_pref[1] for word_pref in words_pref_sorted[:20]]

xs = [i + 0.1 for i, _ in enumerate(words_list)]

# show the graph by using matplotlib
import matplotlib as mpl
mpl.rcParams['font.family'] = 'NanumGothic'

from matplotlib import pyplot as plt
plt.bar(xs, words_pref_list)
plt.ylabel("Preference")
plt.title("Preference of Words Related '" + keyword + "'")

plt.xticks([i + 0.5 for i, _ in enumerate(words_list)], words_list, rotation='vertical')
plt.subplots_adjust(bottom=0.2)
plt.show()