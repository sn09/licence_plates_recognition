"""Model training module."""

import logging
from pathlib import Path

import lightning as L
import mlflow
from hydra import compose, initialize
from omegaconf import OmegaConf

from .helpers.utils import (
    get_callbacks,
    get_dataset,
    get_loggers,
    get_model,
    get_transformers,
)


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def train(
    config_name: str = "training",
    config_path: str | None = None,
):
    """Main train model function.

    Args:
        config_name: config name to use
        config_path: path to config location
    """
    config_path = config_path or Path(__file__).parent / "configs"
    LOGGER.info("Using config_path %s/%s", config_path, config_name)
    with initialize(version_base=None, config_path=config_path):
        config = compose(config_name=config_name, return_hydra_config=True)
        LOGGER.info("Loaded training config:\n%s", OmegaConf.to_yaml(config))

        # Get model
        model = get_model(
            config.model,
            optimizer_cfg=config.optimizer,
            scheduler_cfg=config.scheduler,
            alphabet=config.data.dataset.params.alphabet,
        )
        LOGGER.info("Model is loaded")

        # Get data
        base_path = Path(config.data.path)
        transformers = get_transformers(config.transformers, is_train=True)
        data_module = get_dataset(
            config.data.dataset,
            train_path=base_path / "train",
            test_path=base_path / "test",
            transformers=transformers,
        )
        LOGGER.info("Data module is loaded train_path=%s", base_path / "train")

        # Get callbacks
        callbacks = get_callbacks(config.callbacks)

        # Setup loggers
        loggers_dct = get_loggers(config.loggers)
        loggers = loggers_dct.values()

        # Get trainer
        trainer = L.Trainer(**config.trainer.params, callbacks=callbacks, logger=loggers)

        # Setup mlflow run
        if config.loggers.mlflow.use:
            mlflow.set_tracking_uri(uri=config.loggers.mlflow.params.tracking_uri)
            mlflow.set_experiment(config.loggers.mlflow.params.experiment_name)

            if config.loggers.mlflow.autolog.use:
                LOGGER.info(
                    "Using mlflow autolog, params - %s",
                    config.loggers.mlflow.autolog.params,
                )
                mlflow.pytorch.autolog(**config.loggers.mlflow.autolog.params)

            with mlflow.start_run(run_id=loggers_dct["mlflow"].run_id):
                mlflow.log_params(
                    {
                        "config": config,
                    }
                )
                trainer.fit(model=model, datamodule=data_module)
        else:
            trainer.fit(model=model, datamodule=data_module)

        final_model_path = (
            Path(config.callbacks.model_checkpoint.params.dirpath)
            / f"{config.training_params.model_filename}.ckpt"
        )
        trainer.save_checkpoint(final_model_path)
        LOGGER.info("Saving model to %s", final_model_path)


if __name__ == "__main__":
    train()
