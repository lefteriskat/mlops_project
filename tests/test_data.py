from src.data.data import SpamDatasetDataModule
from src import _PATH_DATA
from hydra import initialize, compose

def test_dataset():
    with initialize(version_base=None, config_path="../config"):
        # config is relative to a module
        config = compose(config_name="config_all.yaml")
    data_module = SpamDatasetDataModule(data_path = _PATH_DATA, config = config)
    data_module.setup()

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()

    total_length = len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset)
    assert total_length == 5572