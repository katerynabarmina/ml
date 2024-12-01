import gensim.downloader as api

model = api.load("word2vec-google-news-300")

sim = model.similarity('flower', 'roses')
print("\nCalculating similarity: 'flower' & 'roses'")
print(sim)