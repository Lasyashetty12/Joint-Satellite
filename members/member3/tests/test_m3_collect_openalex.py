
import json

from members.member3.m3_collect_openalex import (
    validate_works,
    OUTPUT_FILE,
)


def test_sample_file_contains_valid_papers():
    assert OUTPUT_FILE.exists(), (
        "Run m3_collect_openalex.py before running this test."
    )

    with OUTPUT_FILE.open(encoding="utf-8") as file:
        works = json.load(file)

    assert validate_works(works)
    assert len(works) > 0

    for paper in works:
        assert paper["id"]
        assert paper["title"]
        assert paper["publication_year"] is not None
        assert paper["cited_by_count"] is not None
