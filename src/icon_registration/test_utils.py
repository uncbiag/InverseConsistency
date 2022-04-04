
import subprocess
import pathlib
import sys

TEST_DATA_DIR = pathlib.Path(__file__).parent.parent.parent / "test_files"


def download_test_data():
    subprocess.run(["girder-client",  "--api-url", "https://data.kitware.com/api/v1", "localsync", "61d3a99d4acac99f429277d7", str(TEST_DATA_DIR)], stdout=sys.stdout)
