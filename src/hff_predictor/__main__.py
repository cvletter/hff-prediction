import argparse
from hff_predictor.data.prepare import init_prepare_data
from hff_predictor.model.fit import init_train
from hff_predictor.predict.make import init_predict
from hff_predictor.predict.test import init_test
from hff_predictor.evaluation.evaluate import init_evaluate
from hff_predictor.config.folder_structure import init_folder_setup
import hff_predictor.config.column_names as cn

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
        "evaluate": [init_evaluate],
        "setup_folder": [init_folder_setup]
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
        default=cn.DEFAULT_PRED_DATE
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
        default=2
    )

    parser.add_argument(
        "--su_member",
        "-f",
        type=str,
        default=None
    )

    parser.add_argument(
        "--summary",
        "-s",
        type=str,
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="find"

    )

    arguments, _ = parser.parse_known_args()

    for method in mode_methods[arguments.mode]:
        if arguments.mode in ["predict"]:
            method(arguments.date, arguments.window, arguments.reload, arguments.su_member)
        elif arguments.mode in ["test"]:
            method(arguments.date, arguments.periods)
        elif arguments.mode in ["evaluate"]:
            method(arguments.summary)
        elif arguments.mode in ["setup_folder"]:
            method(arguments.output)
        else:
            method()

    LOGGER.info("Ended")


if __name__ == "__main__":
    main()
