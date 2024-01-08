"""CLI interface for model training and inference."""

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from recognition_app.infer import infer
from recognition_app.train import train


def _setup_parser(parser: ArgumentParser) -> None:
    subparsers = parser.add_subparsers(help="choose comand: (build, query)")

    # Model training
    train_parser = subparsers.add_parser(
        "train",
        help="tool for training model",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    train_parser.add_argument(
        "--config-name",
        help="config name to use",
        default="training",
    )
    train_parser.add_argument(
        "--config-path",
        help="path to config file",
        default="../configs/",
    )
    train_parser.set_defaults(callback=train)

    # Model inference
    infer_parser = subparsers.add_parser(
        "infer",
        help="tool for model inference",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    infer_parser.add_argument(
        "--config-name",
        help="config name to use",
        default="inference",
    )
    infer_parser.add_argument(
        "--config-path",
        help="path to config file",
        default="../configs/",
    )
    infer_parser.set_defaults(callback=infer)


def main():
    """Main CLI interface function."""

    parser = ArgumentParser(
        prog="model-cli",
        description="CLI tool for licence plates recognition model training and inference",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    _setup_parser(parser)
    arguments = parser.parse_args()
    arguments.callback(
        config_name=arguments.config_name, config_path=arguments.config_path
    )


if __name__ == "__main__":
    main()
