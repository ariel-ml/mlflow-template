import pandas as pd
from sklearn import datasets
from libs import logger, paths

LOG = logger.getLogger(__name__)


def process():
    LOG.info("Downloading data...")
    data = datasets.load_iris(as_frame=True)
    X = data["data"]
    y = data["target"].astype("float32")
    df = pd.concat([X, y], axis=1)
    df.to_parquet(paths.RAW_DATA_FILE)
    LOG.info("Data downloaded and saved to %s", paths.RAW_DATA_FILE)


if __name__ == "__main__":
    process()
