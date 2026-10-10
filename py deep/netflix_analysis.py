"""
Netflix Movies and TV Shows Analysis
Mini-project using Pandas, NumPy, Matplotlib and Seaborn.

Dataset expected: netflix_titles.csv
Common public dataset fields include:
show_id, type, title, director, cast, country, date_added, release_year,
rating, duration, listed_in, description

Run:
    pip install -r requirements.txt
    python netflix_analysis.py

Place netflix_titles.csv in the same folder as this script.
The script creates a 'outputs' folder with charts and a text summary.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_FILE = Path("netflix_titles.csv")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", palette="Set2")

def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "Could not find netflix_titles.csv. Download the Netflix Movies and TV Shows "
            "dataset (commonly available on Kaggle) and put the CSV beside this script."
        )

    df = pd.read_csv(DATA_FILE)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    required = {"type", "title", "release_year", "rating", "listed_in"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    # Data cleaning
    df = df.drop_duplicates()
    df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce")
    df["rating"] = df["rating"].fillna("Not Rated")
    df["type"] = df["type"].fillna("Unknown")
    df["listed_in"] = df["listed_in"].fillna("Unknown")
    df["title"] = df["title"].fillna("Untitled")
    clean = df.dropna(subset=["release_year"]).copy()
    clean["release_year"] = clean["release_year"].astype(int)

    # Use NumPy to calculate a compact numerical summary.
    years = clean["release_year"].to_numpy(dtype=float)
    year_summary = {
        "minimum_release_year": int(np.min(years)) if len(years) else None,
        "maximum_release_year": int(np.max(years)) if len(years) else None,
        "median_release_year": float(np.median(years)) if len(years) else None,
        "mean_release_year": float(np.mean(years)) if len(years) else None,
    }

    # Type distribution
    type_counts = clean["type"].value_counts()
    plt.figure(figsize=(7, 5))
    ax = sns.barplot(x=type_counts.index, y=type_counts.values)
    ax.set_title("Netflix Titles by Content Type")
    ax.set_xlabel("Content type")
    ax.set_ylabel("Number of titles")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "01_content_type.png", dpi=200)
    plt.close()

    # Release year trend: count titles by release year
    yearly = clean.groupby(["release_year", "type"]).size().reset_index(name="count")
    plt.figure(figsize=(11, 6))
    sns.lineplot(data=yearly, x="release_year", y="count", hue="type", marker="o")
    plt.title("Netflix Titles by Release Year and Type")
    plt.xlabel("Release year")
    plt.ylabel("Number of titles in dataset")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "02_release_year_trend.png", dpi=200)
    plt.close()

    # Ratings
    rating_counts = clean["rating"].value_counts().head(12)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=rating_counts.values, y=rating_counts.index)
    plt.title("Most Common Content Ratings")
    plt.xlabel("Number of titles")
    plt.ylabel("Rating")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "03_content_ratings.png", dpi=200)
    plt.close()

    # Genres: listed_in may contain multiple comma-separated categories.
    genre_series = clean["listed_in"].str.split(",").explode().str.strip()
    genre_series = genre_series[genre_series.ne("") & genre_series.ne("Unknown")]
    genre_counts = genre_series.value_counts().head(15)
    plt.figure(figsize=(10, 7))
    sns.barplot(x=genre_counts.values, y=genre_counts.index)
    plt.title("Top 15 Genres / Categories")
    plt.xlabel("Number of category assignments")
    plt.ylabel("Genre / category")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "04_top_genres.png", dpi=200)
    plt.close()

    # Genre by content type (top 10 overall categories)
    top_genres = genre_counts.head(10).index
    genre_rows = clean[["type", "listed_in"]].copy()
    genre_rows["genre"] = genre_rows["listed_in"].str.split(",")
    genre_rows = genre_rows.explode("genre")
    genre_rows["genre"] = genre_rows["genre"].str.strip()
    genre_rows = genre_rows[genre_rows["genre"].isin(top_genres)]
    genre_type = genre_rows.groupby(["genre", "type"]).size().reset_index(name="count")
    plt.figure(figsize=(11, 7))
    sns.barplot(data=genre_type, y="genre", x="count", hue="type")
    plt.title("Top Genres by Content Type")
    plt.xlabel("Number of category assignments")
    plt.ylabel("Genre / category")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "05_genre_by_type.png", dpi=200)
    plt.close()

    # Text summary: real values from the CSV, not hard-coded claims.
    total = len(clean)
    summary_lines = [
        "NETFLIX DATA ANALYSIS SUMMARY",
        "============================",
        f"Rows after duplicate removal and release-year cleaning: {total}",
        "",
        "Content type counts:",
        type_counts.to_string(),
        "",
        "Top content ratings:",
        clean["rating"].value_counts().head(10).to_string(),
        "",
        "Top genre/category assignments:",
        genre_counts.head(10).to_string(),
        "",
        "Release year statistics (titles with a valid release year):",
        str(year_summary),
        "",
        "Charts saved in the outputs folder.",
        "Note: genre counts are category assignments; a single title may belong to multiple genres."
    ]
    (OUT_DIR / "analysis_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    print("\n".join(summary_lines))
    print(f"\nDone. Charts and summary saved to: {OUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
