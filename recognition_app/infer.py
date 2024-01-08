"""Model inference module."""

import logging
from pathlib import Path

import lightning as L
import pandas as pd
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

from .helpers.utils import decode, get_dataset, get_transformers, import_object


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def infer(
    config_name: str = "inference",
    config_path: str | None = None,
):
    """Main model inference function.

    Args:
        config_name: config name to use
        config_path: path to config location
    """
    config_path = config_path or Path(__file__).parent / "configs"
    with initialize(version_base=None, config_path=config_path):
        config: DictConfig = compose(config_name=config_name, return_hydra_config=True)
        LOGGER.info("Loaded training config:\n%s", OmegaConf.to_yaml(config))

        # Load module from checkpoint
        model_cls = import_object(config.model.model.class_factory)
        model = model_cls.load_from_checkpoint(config.inference_params.checkpoint_path)
        LOGGER.info(
            "Loading model from checkpoint: %s", config.inference_params.checkpoint_path
        )

        # Get data
        base_path = Path(config.data.path)
        transformers = get_transformers(config.model.transformers, is_train=False)
        data_module = get_dataset(
            config.data.dataset,
            train_path=base_path / "train",
            test_path=base_path / "test",
            transformers=transformers,
        )
        LOGGER.info("Data module is loaded test_path=%s", base_path / "test")

        # Get predictions
        trainer = L.Trainer()
        predictions = trainer.predict(model, datamodule=data_module)
        predictions = decode(predictions, model.alphabet)

        # Prepare output dataframe
        final_df = pd.DataFrame(
            {
                "filepath": data_module.test_images,
                "label": predictions,
            }
        )
        final_df.to_csv(config.inference_params.output_path, index=False)
        LOGGER.info("Save prediction to %s.", config.inference_params.output_path)
