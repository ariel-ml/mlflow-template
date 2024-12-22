from libs.data_etl import data_utils, paths
from sklearn.model_selection import train_test_split
from libs import logger


LOG = logger.getLogger(__name__)


def process():
    LOG.info("Splitting data...")
    X, y, columns = data_utils.load_data(paths.RAW_DATA_FILE)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    data_utils.save_data(
        data_utils.concat(X_train, y_train, columns=columns),
        paths.SPLIT_TRAIN_FILE,
    )
    data_utils.save_data(
        data_utils.concat(X_test, y_test, columns=columns),
        paths.SPLIT_TEST_FILE,
    )

    LOG.info("Data split and saved to %s", paths.SPLIT_DATA_DIR)


if __name__ == "__main__":
    process()
