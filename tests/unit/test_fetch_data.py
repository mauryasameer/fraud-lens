import pytest

import scripts.fetch_data as fetch_data


def test_fetch_creditcard_data_exits_when_credentials_missing(mocker, tmp_path):
    mocker.patch.object(fetch_data, "KAGGLE_CREDENTIALS", tmp_path / "missing.json")
    mock_run = mocker.patch("subprocess.run")

    with pytest.raises(SystemExit) as exc_info:
        fetch_data.fetch_creditcard_data()

    assert exc_info.value.code == 1
    mock_run.assert_not_called()


def test_fetch_creditcard_data_downloads_and_extracts(mocker, tmp_path):
    creds = tmp_path / "kaggle.json"
    creds.write_text("{}")
    data_dir = tmp_path / "data"
    mocker.patch.object(fetch_data, "KAGGLE_CREDENTIALS", creds)
    mocker.patch.object(fetch_data, "DATA_DIR", data_dir)
    mock_run = mocker.patch("subprocess.run")
    mock_zip = mocker.patch("zipfile.ZipFile")
    mocker.patch("pathlib.Path.unlink")

    fetch_data.fetch_creditcard_data()

    mock_run.assert_called_once()
    assert mock_run.call_args.kwargs["check"] is True
    mock_zip.assert_called_once()
