import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Return the most 10 similar articles
def get_recommendations(data, indices, title, cosine_sim):
    # Get the index of the article matching the title
    idx = indices[title]

    # Compute the pairwsie similarity scores of all articles and sort them based on similarity
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the top 10 articles
    sim_scores = sim_scores[1:11]
    article_indices = [i[0] for i in sim_scores]

    return data['title'].iloc[article_indices]


def main():
    data = pd.read_csv('medium.csv', low_memory=False)

    # Create a TF-IDF vectorizer and remove stopwords and NaN
    tfidf = TfidfVectorizer(stop_words='english')
    data['text'] = data['text'].fillna('')

    # Construct TF-IDF matrix and cosine similarity matrix
    tfidf_matrix = tfidf.fit_transform(data['text'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    indices = pd.Series(data.index, index=data['title']).drop_duplicates()

    print(get_recommendations(data, indices, 'This the music you should run to, because science', cosine_sim))

if  __name__ =='__main__':
    main()
