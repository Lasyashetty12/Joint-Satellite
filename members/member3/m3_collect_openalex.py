
import json
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

OPENALEX_URL = "https://api.openalex.org/works"
OUTPUT_FILE = Path(__file__).parent / "data" / "sample_openalex.json"

FIELDS = (
    "id,doi,title,publication_year,"
    "cited_by_count,authorships"
)


def fetch_works(query="artificial intelligence", per_page=3):
    """Fetch research-paper metadata from OpenAlex."""
    params = {
        "search": query,
        "per-page": per_page,
        "select": FIELDS,
    }

    # Optional credentials, if your OpenAlex account requires them.
    api_key = os.getenv("OPENALEX_API_KEY")
    mailto = os.getenv("OPENALEX_MAILTO")

    if api_key:
        params["api_key"] = api_key
    if mailto:
        params["mailto"] = mailto

    response = requests.get(
        OPENALEX_URL,
        params=params,
        timeout=30,
    )
    response.raise_for_status()

    payload = response.json()
    works = payload.get("results", [])

    if not isinstance(works, list):
        raise ValueError("OpenAlex returned an invalid results format.")

    return works


def validate_works(works):
    """Check that each paper has the required metadata."""
    required_fields = [
        "id",
        "title",
        "publication_year",
        "cited_by_count",
    ]

    if not works:
        raise ValueError("No research papers were returned.")

    for index, paper in enumerate(works):
        missing = [
            field for field in required_fields
            if paper.get(field) is None
        ]

        if missing:
            raise ValueError(
                f"Paper {index} is missing fields: {missing}"
            )

    return True


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    print("Fetching research papers from OpenAlex...")

    works = fetch_works(per_page=3)
    validate_works(works)

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(works, file, indent=2, ensure_ascii=False)

    print(f"Successfully collected {len(works)} papers.")
    print(f"Saved data to: {OUTPUT_FILE}")
    print("Validation passed.")


if __name__ == "__main__":
    main()
