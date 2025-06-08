# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Script to train/fine-tune a Contextual ASR model (e.g., ContextualEncDecRNNTModel).
This script directly instantiates the contextual model and can load weights
from a base non-contextual ASR model for fine-tuning.
"""

import torch
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

from nemo.collections.asr.models import ASRModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg


@hydra_runner(config_path="conf/fastconformer", config_name="contextual_fast-conformer_transducer_bpe")
def main(cfg: DictConfig) -> None:
    logging.info(f"Hydra config:\n{OmegaConf.to_yaml(cfg)}")

    trainer = pl.Trainer(**resolve_trainer_cfg(cfg.trainer))
    exp_manager(trainer, cfg.get("exp_manager", None))

    logging.info(f"Instantiating model as per config: {cfg.model._target_}")
    asr_model = ASRModel.from_config_dict(cfg.model, trainer=trainer)

    if cfg.model.get('fine_tune_from_nemo_model', None) is not None:
        base_model_path = cfg.model.fine_tune_from_nemo_model
        logging.info(f"Attempting to load weights from base model: {base_model_path}")
        try:
            logging.info(f"Loading base model from: {base_model_path}")
            base_model_instance = ASRModel.restore_from(base_model_path, map_location='cpu', trainer=trainer)
            base_weights = base_model_instance.state_dict()

            asr_model.load_state_dict(base_weights, strict=False)
            logging.info(f"Successfully loaded weights from {base_model_path} into the contextual model (strict=False).")
            del base_model_instance
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logging.error(f"Could not load or transfer weights from {base_model_path}: {e}")
            logging.warning("Proceeding with randomly initialized weights for the entire model or parts not loaded.")

    asr_model.setup_training_data(cfg.model.train_ds)
    asr_model.setup_multiple_validation_data(cfg.model.validation_ds)
    if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None:
        asr_model.setup_multiple_test_data(cfg.model.test_ds)

    if cfg.model.get('optim', None) is not None:
        asr_model.setup_optimization(cfg.model.optim)
    else:
        logging.warning("No optimizer configuration found in cfg.model.optim. Please ensure it's defined.")

    logging.info("Starting training...")
    trainer.fit(asr_model)

    if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None and trainer.is_global_zero:
        if torch.cuda.is_available():
            asr_model.cuda()
        logging.info("Starting testing...")
        trainer.test(asr_model, ckpt_path=None)

if __name__ == '__main__':
    main()
