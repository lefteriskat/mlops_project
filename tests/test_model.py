import torch
from hydra import compose, initialize

from src.data.data import SpamDatasetDataModule
from src.models.model import AwesomeSpamClassificationModel


def test_model():
    with initialize(version_base=None, config_path="../config"):
        config = compose(config_name="default_config.yaml")

    data_module = SpamDatasetDataModule(config=config)
    data_module.setup()

    test_loader = data_module.test_dataloader()

    model = AwesomeSpamClassificationModel(config=config)

    x = next(iter(test_loader))

    (logits,) = model(x)

    assert logits.shape == torch.Size(
        [config.train.batch_size, config.model.output_size]
    )
