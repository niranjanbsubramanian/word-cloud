import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from PIL import Image

# text from wikipedia
text = 'Supervised learning is the machine learning task of learning a function that maps an input to \
        an output based on example input-output pairs.[1] It infers a function from labeled training data \
        consisting of a set of training examples.[2] In supervised learning, each example is a pair consisting \
        of an input object (typically a vector) and a desired output value (also called the supervisory signal). \
        A supervised learning algorithm analyzes the training data and produces an inferred function, which can \
        be used for mapping new examples. An optimal scenario will allow for the algorithm to correctly determine \
        the class labels for unseen instances. This requires the learning algorithm to generalize from the training \
        data to unseen situations in a "reasonable" way (see inductive bias).'

mask_img = np.array(Image.open("photo.jpg"))

wordcloud = WordCloud(background_color='white', mask=mask_img, \
        stopwords=set(STOPWORDS)).generate(text)

wordcloud.to_file('word_cloud.png')
import matplotlib.pyplot as plt
plt.imshow(wordcloud.recolor(random_state=2020))
plt.title('Word Cloud')
plt.axis("off")
plt.show()