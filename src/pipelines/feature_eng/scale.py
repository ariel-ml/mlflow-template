import joblib
from libs.data_etl import data_utils, paths
from sklearn.preprocessing import StandardScaler
from libs import logger


LOG = logger.getLogger(__name__)


def process():
    LOG.info("Scaling data...")
    X_train, y_train, columns = data_utils.load_data(paths.SPLIT_TRAIN_FILE)
    X_test, y_test, _ = data_utils.load_data(paths.SPLIT_TEST_FILE)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    data_utils.save_data(
        data_utils.concat(X_train_scaled, y_train, columns=columns),
        paths.TRAIN_DATA_FILE,
    )
    data_utils.save_data(
        data_utils.concat(X_test_scaled, y_test, columns=columns),
        paths.TEST_DATA_FILE,
    )

    joblib.dump(scaler, paths.MODEL_DATA_DIR / "scaler.pkl")

    LOG.info("Data scaled and saved to %s", paths.MODEL_INPUT_DATA_DIR)


if __name__ == "__main__":
    process()
