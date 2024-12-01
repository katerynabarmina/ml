import gensim.downloader as api

model = api.load("word2vec-google-news-300")
word_to_search = "heart"
similar_words = model.most_similar(word_to_search, topn=5)
print(f"Top 5 similar words to '{word_to_search}':", similar_words)