import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from text_cleaner import clean_text


def match_resumes(resume_file, job_file):
    # Get project root directory
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Build correct file paths
    resume_path = os.path.join(BASE_DIR, resume_file)
    job_path = os.path.join(BASE_DIR, job_file)

    # Load data
    resumes = pd.read_csv(resume_path)
    with open(job_path, "r", encoding="utf-8") as f:
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

    print("\n=== Resume Ranking Results ===\n")
    for idx, row in results.iterrows():
        print(
            f"Rank {idx + 1}: {row['resume']} "
            f"(Score: {row['match_score']:.4f})"
        )
