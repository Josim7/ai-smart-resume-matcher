import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from text_cleaner import clean_text


def match_resumes(resume_file, job_file):
    # Load data
    resumes = pd.read_csv(resume_file)
    with open(job_file, "r") as f:
        job_description = f.read()

    # Clean text
    resumes["cleaned"] = resumes["resume"].apply(clean_text)
    job_cleaned = clean_text(job_description)

    # Vectorization
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(
        resumes["cleaned"].tolist() + [job_cleaned]
    )

    # Similarity
    scores = cosine_similarity(vectors[:-1], vectors[-1])

    resumes["match_score"] = scores
    return resumes.sort_values(by="match_score", ascending=False)


if __name__ == "__main__":
    results = match_resumes(
        "data/resumes.csv",
        "data/job_description.txt"
    )
    print(results)
