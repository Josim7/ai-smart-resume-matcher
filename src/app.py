import sys
import os
import pandas as pd

# Add src directory to path
sys.path.append(os.path.dirname(__file__))

from matcher import rank_resumes


def main():
    resumes_path = os.path.join("data", "resumes.csv")
    job_desc_path = os.path.join("data", "job_description.txt")

    results = rank_resumes(resumes_path, job_desc_path)

    print("\n=== Resume Ranking Results ===\n")
    for idx, row in results.iterrows():
        print(f"Rank {idx + 1}: {row['resume']} (Score: {row['score']:.4f})")


if __name__ == "__main__":
    main()
