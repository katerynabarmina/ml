import gensim.downloader as api

model = api.load("word2vec-google-news-300")

doesnt_matched_word = model.doesnt_match(
    'sunflower roses dandelions learning'.split(),
)
print(doesnt_matched_word)