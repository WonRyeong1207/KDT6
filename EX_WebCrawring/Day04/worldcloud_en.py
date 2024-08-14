from wordcloud import WordCloud
from wordcloud import STOPWORDS
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

text = open('./data/alice.txt').read()
STOPWORDS.add('said')
print()
print('STOPWORDS: ', STOPWORDS)
print()

img_mask = np.array(Image.open('./data/cloud.png'))

wc = WordCloud(width=400, height=400, background_color='white',
                      max_font_size=200, stopwords=STOPWORDS, repeat=True,
                      colormap='inferno', mask=img_mask)

print(type(wc))
print()
print(type(text))
wordcloud = wc.generate(text)


print(wordcloud.words_)
print()
wordcloud.to_file('./data/alice.jpg')
plt.figure(figsize=(10,8))
plt.axis('off')
plt.imshow(wordcloud)
plt.show()