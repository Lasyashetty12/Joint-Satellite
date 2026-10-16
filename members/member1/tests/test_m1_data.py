import importlib.util
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "m1_dataset_inventory.py"
)
SPEC = importlib.util.spec_from_file_location("m1_dataset_inventory", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_inventory_finds_supported_image(tmp_path):
    image = tmp_path / "sample.tif"
    image.write_bytes(b"tiny-test")
    (tmp_path / "notes.txt").write_text("not an image")

    results = MODULE.inventory_dataset(tmp_path)

    assert len(results) == 1
    assert results[0][0].name == "sample.tif"
    assert results[0][1] == len(b"tiny-test")


def test_inventory_includes_nested_files(tmp_path):
    nested = tmp_path / "subfolder"
    nested.mkdir()
    (nested / "sample.png").write_bytes(b"png")

    results = MODULE.inventory_dataset(tmp_path)

    assert len(results) == 1
    assert results[0][0].name == "sample.png"


def test_missing_folder_raises_error(tmp_path):
    missing = tmp_path / "does_not_exist"

    try:
        MODULE.inventory_dataset(missing)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("Expected FileNotFoundError")
