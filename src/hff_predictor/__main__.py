import argparse
from hff_predictor.data.prepare import init_prepare_data
from hff_predictor.model.fit import init_train
from hff_predictor.predict.make import init_predict
from hff_predictor.predict.test import init_test
from hff_predictor.evaluation.evaluate import init_evaluate

import pkg_resources
import logging.config

logging.config.fileConfig(
    pkg_resources.resource_filename(f"{__package__}.resources", "logger.ini"), disable_existing_loggers=False
)

LOGGER = logging.getLogger(__name__)


def main():
    LOGGER.info('Started')

    mode_methods = {
        "prepare": [init_prepare_data],
        "train": [init_train],
        "predict": [init_predict],
        "test": [init_test],
        "evaluate": [init_evaluate]
    }

    parser = argparse.ArgumentParser(description="HFF predictor")
    parser.add_argument(
        "--mode",
        "-m",
        choices=mode_methods.keys(),
    )

    parser.add_argument(
        "--date",
        "-d",
        type=str,
        required=True
    )

    parser.add_argument(
        "--window",
        "-w",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--reload",
        "-r",
        action='store_true',
        default=False,
    )

    parser.add_argument(
        "--periods",
        "-p",
        type=int,
    )

    parser.add_argument(
        "--summary",
        "-s",
        type=str,
    )

    arguments, _ = parser.parse_known_args()

    for method in mode_methods[arguments.mode]:
        if arguments.mode in ["predict"]:
            method(arguments.date, arguments.window, arguments.reload)
        elif arguments.mode in ["test"]:
            method(arguments.date, arguments.periods)
        elif arguments.mode in ["evaluate"]:
            method(arguments.summary)
        else:
            method()

    LOGGER.info("Ended")


if __name__ == "__main__":
    main()
