import json
import pytest
import pandas as pd
from actantial.io import parse_json, load_annotations, load_actors


class TestParseJson:

    def test_valid_json(self):
        result = parse_json('{"Subject": "Alice", "Object": "Bob"}')
        assert result == {"Subject": "Alice", "Object": "Bob"}

    def test_json_embedded_in_text(self):
        result = parse_json('Here is the answer: {"Subject": "Alice"} as requested.')
        assert result == {"Subject": "Alice"}

    def test_no_json_returns_empty_dict(self):
        result = parse_json("There is no JSON here.")
        assert result == {}

    def test_multiple_json_objects_returns_first(self):
        result = parse_json('{"Subject": "Alice"} and {"Object": "Bob"}')
        assert result == {"Subject": "Alice"}

    def test_malformed_json_returns_empty_dict(self):
        result = parse_json('{"Subject": "Alice"')
        assert result == {}

    def test_empty_json_object(self):
        result = parse_json("{}")
        assert result == {}

    def test_values_with_lists(self):
        result = parse_json('{"Subject": ["Alice", "Bob"]}')
        assert result == {"Subject": ["Alice", "Bob"]}


class TestLoadActors:

    def test_extracts_actants_from_files(self, tmp_path):
        annotation = {
            "Subject": "Alice",
            "Object": "X",
            "Sender": "Bob",
            "Receiver": "Carl",
            "Helper": "",
            "Opponent": "",
        }
        f = tmp_path / "row1.txt"
        f.write_text(json.dumps(annotation))

        data = pd.DataFrame({"id": ["row1"], "file_name": [str(f)]})
        result = load_actors(data)

        assert result.loc[0, "Subject"] == "Alice"
        assert result.loc[0, "Object"] == "X"

    def test_select_actor_first(self, tmp_path):
        f = tmp_path / "row1.txt"
        f.write_text(json.dumps({"Subject": ["Alice", "Bob"]}))

        data = pd.DataFrame({"id": ["row1"], "file_name": [str(f)]})
        result = load_actors(data, select_actor="first")

        assert result.loc[0, "Subject"] == "Alice"

    def test_select_actor_combine(self, tmp_path):
        f = tmp_path / "row1.txt"
        f.write_text(json.dumps({"Subject": ["Alice", "Bob"]}))

        data = pd.DataFrame({"id": ["row1"], "file_name": [str(f)]})
        result = load_actors(data, select_actor="combine")

        assert result.loc[0, "Subject"] == "Alice, Bob"

    def test_missing_file_skipped(self):
        data = pd.DataFrame({"id": ["row1"], "file_name": [None]})
        result = load_actors(data)
        assert result.loc[0, "Subject"] is None

    def test_invalid_select_actor_raises(self, tmp_path):
        f = tmp_path / "row1.txt"
        f.write_text(json.dumps({"Subject": "Alice"}))

        data = pd.DataFrame({"id": ["row1"], "file_name": [str(f)]})
        with pytest.raises(ValueError, match="select_actor"):
            load_actors(data, select_actor="invalid")


class TestLoadAnnotations:

    def test_loads_annotations_by_id(self, tmp_path):
        (tmp_path / "row1.txt").write_text(
            json.dumps({"Subject": "Alice", "Object": "X"})
        )
        (tmp_path / "row2.txt").write_text(
            json.dumps({"Subject": "Bob", "Object": "Y"})
        )

        data = pd.DataFrame({"id": ["row1", "row2"], "text": ["a", "b"]})
        result = load_annotations(data, label_folder=str(tmp_path))

        assert result.loc[0, "Subject"] == "Alice"
        assert result.loc[1, "Subject"] == "Bob"

    def test_missing_label_folder_raises(self):
        data = pd.DataFrame({"id": ["row1"]})
        with pytest.raises(KeyError):
            load_annotations(data, label_folder="/nonexistent/path")

    def test_missing_id_column_raises(self, tmp_path):
        data = pd.DataFrame({"text": ["hello"]})
        with pytest.raises(KeyError, match="id"):
            load_annotations(data, label_folder=str(tmp_path))

    def test_missing_annotation_files_warns(self, tmp_path):
        data = pd.DataFrame({"id": ["missing_id"], "text": ["a"]})
        with pytest.warns(UserWarning):
            load_annotations(data, label_folder=str(tmp_path))
