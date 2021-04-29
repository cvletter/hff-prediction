import argparse

from hff_predictor.data.prepare import init_prepare_data
from hff_predictor.evaluation.descriptive_analysis import init_descriptive_analysis
from hff_predictor.evaluation.prediction import init_evaluate_prediction
from hff_predictor.features.create import init_create_features
from hff_predictor.model.fit import init_train
from hff_predictor.predict.make import init_predict
from hff_predictor.predict.setup import init_setup_prediction
from hff_predictor.predict.test import init_test

import logging


def main():
    logging.basicConfig(filename='test.log', level=logging.INFO)
    logging.info('Started')

    mode_methods = {
        "prepare": [init_prepare_data],
        "train": [init_train],
        "predict": [init_predict],
        "test": [init_test],
        "evaluate": [init_evaluate_prediction, init_descriptive_analysis],
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
    )

    parser.add_argument(
        "--window",
        "-w",
        type=int,
    )

    parser.add_argument(
        "--reload",
        "-r",
        type=str,
    )

    parser.add_argument(
        "--periods",
        "-p",
        type=int,
    )

    arguments, _ = parser.parse_known_args()

    for method in mode_methods[arguments.mode]:
        if arguments.mode in ["predict"]:
            method(arguments.date, arguments.window, arguments.reload)
        elif arguments.mode in ["test"]:
            method(arguments.date, arguments.periods)
        else:
            method()

    logging.info("Ended")
if __name__ == "__main__":
    main()
