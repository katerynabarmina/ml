import gensim.downloader as api

model = api.load("word2vec-google-news-300")
similar_words = model.most_similar(
    positive=['programming', 'laptop'],
    topn=5,
)
for inx, sw in enumerate(similar_words):
    print(f"{inx}: {sw}")