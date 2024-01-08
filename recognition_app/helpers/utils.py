"""Module with different utils for prediction, decoding, saving and loading models."""
import logging
from importlib import import_module

import lightning as L
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def import_object(object_path: str):
    """Import object by path string.

    Args:
        - object_path: path to import (format module.submodule_1.submodule_2.object)
    """
    module_name, class_name = object_path.rsplit(".", maxsplit=1)
    return getattr(import_module(module_name), class_name)


def pred_to_string(prediction: torch.Tensor, alphabet: str) -> str:
    """Decode single prediction to string format.

    Args:
        - prediction: single prediction Tensor
        - alphabet: allowed alphabet

    Returns:
        decoded string
    """
    encoded_idx = []
    for pred in prediction:
        encoded_idx.append(np.argmax(pred) - 1)

    encoded_letters = []
    for i, pred in enumerate(encoded_idx):
        if pred == -1:
            continue
        if not encoded_letters:
            encoded_letters.append(alphabet[pred])
            continue
        if pred != encoded_idx[i - 1]:
            encoded_letters.append(alphabet[pred])
    output_string = "".join(encoded_letters)

    return output_string


def decode(predictions: torch.Tensor, alphabet: str) -> list[str]:
    """Decode raw predictions to string format.

    Args:
        - predictions: predictions Tensor
        - alphabet: allowed alphabet

    Returns:
        list of decoded strings
    """
    predictions = predictions.permute(1, 0, 2).cpu().data.numpy()
    outputs = []
    for pred in predictions:
        outputs.append(pred_to_string(pred, alphabet))
    return outputs


def get_callbacks(config: DictConfig) -> list[L.Callback]:
    """Get Lightning callbacks from config.

    Args:
        - config: hydra config

    Returns:
        List of Lightning callbacks
    """
    callbacks_lst = []
    for name, cfg in config.items():
        if cfg.use:
            params = cfg.params if hasattr(cfg, "params") else dict()
            callback_cls = import_object(cfg.class_factory)
            callbacks_lst.append(callback_cls(**params))

            LOGGER.info("Using callback %s, params - %s.", (name, params))

    return callbacks_lst


def get_loggers(config: DictConfig):
    """Get Lightning callbacks from config.

    Args:
        - config: hydra config

    Returns:
        List of Lightning callbacks
    """
    loggers = {}
    for name, cfg in config.items():
        if cfg.use:
            params = cfg.params if hasattr(cfg, "params") else dict()
            logger_cls = import_object(cfg.class_factory)
            loggers[name] = logger_cls(**params)
            LOGGER.info("Using logger %s, params - %s.", (name, params))

    return loggers


def get_model(
    config: DictConfig, optimizer_cfg: DictConfig, scheduler_cfg: DictConfig
) -> L.LightningModule:
    """Get Lightning model from config.

    Args:
        - config: hydra config

    Returns:
        Lightning model
    """
    model_cls = import_object(config.class_factory)
    return model_cls(
        optimizer_config=optimizer_cfg, scheduler_config=scheduler_cfg, **config.params
    )


def get_dataset(
    config: DictConfig,
    train_path: str,
    test_path: str,
    transformers: list[nn.Module],
) -> L.LightningDataModule:
    """Get Lightning dataset from config.

    Args:
        - config: hydra config
        - train_path: path to train dataset
        - test_path: path to test dataset

    Returns:
        Lightning dataset
    """
    dataset_cls = import_object(config.class_factory)
    return dataset_cls(
        train_path=train_path,
        test_path=test_path,
        transformers=transformers,
        **config.params,
    )


def get_optimizer(config: DictConfig, model_params: dict) -> torch.optim.Optimizer:
    """Get torch optimizer from config.

    Args:
        - config: hydra config
        - model_params: model learning parameters

    Returns:
        Torch optimizer
    """
    for name, cfg in config.items():
        if cfg.use:
            LOGGER.info("Using optimizer %s, params - %s.", (name, config.params))
            optimizer_cls = import_object(config.class_factory)
            return optimizer_cls(model_params, **config.params)
    raise ValueError("No optimizers provided in config")


def get_scheduler(
    config: DictConfig, optimizer: torch.optim.Optimizer
) -> torch.optim.lr_scheduler.LRScheduler:
    """Get torch scheduler from config.

    Args:
        - config: hydra config
        - model_params: used optimizer

    Returns:
        Torch scheduler
    """
    for name, cfg in config.items():
        if cfg.use:
            LOGGER.info("Using optimizer %s, params - %s.", (name, config.params))
            scheduler_cls = import_object(config.class_factory)
            return scheduler_cls(optimizer, **config.params)


def get_transformers(config: DictConfig, is_train: bool = True):
    """Get image transformers from config.

    Args:
        - config: hydra config

    Returns:
        List of image transformers
    """
    transformers = []
    for name, cfg in config.items():
        if cfg.use_train and is_train or cfg.use_test and not is_train:
            LOGGER.info("Using transformer %s, params - %s.", (name, config.params))
            transformer_cls = import_object(config.class_factory)
            transformers.append(transformer_cls(**config.params))
    return transformers
