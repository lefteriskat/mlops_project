import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch import optim
from transformers import BertForSequenceClassification
import os

class AwesomeSpamClassificationModel(pl.LightningModule):
    def __init__(self, config: DictConfig):
        super(AwesomeSpamClassificationModel, self).__init__()
        self.config = config
        self.model = BertForSequenceClassification.from_pretrained(
            config.model.pretrained_model,
            torchscript=True,
            num_labels=config.model.output_size,
            output_attentions=False,
            output_hidden_states=False,
        )
        self.save_hyperparameters()

    def forward(self, batch):
        input_ids, attention_mask, _ = batch
        return self.model(
            input_ids=input_ids, token_type_ids=None, attention_mask=attention_mask
        )

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, target = batch
        (train_loss, logits) = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=None,
            labels=target,
        )
        preds = torch.argmax(logits, dim=1)
        correct = (preds == target).sum()
        accuracy = correct / len(target)
        self.log("train_loss", train_loss, prog_bar=True)
        self.log("train_accuracy", accuracy, prog_bar=True)
        return {"loss": train_loss, "preds": preds, "labels": target}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        input_ids, attention_mask, target = batch
        (validation_loss, logits) = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=None,
            labels=target,
        )
        preds = torch.argmax(logits, dim=1)
        correct = (preds == target).sum()
        accuracy = correct / len(target)
        self.log("val_loss", validation_loss, prog_bar=True)
        self.log("val_accuracy", accuracy, prog_bar=True)
        return {"loss": validation_loss, "preds": preds, "labels": target}

    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, target = batch
        (test_loss, logits) = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=None,
            labels=target,
        )
        preds = torch.argmax(logits, dim=1)
        correct = (preds == target).sum()
        accuracy = correct / len(target)
        self.log("test_loss", test_loss, prog_bar=True)
        self.log("test_accuracy", accuracy, prog_bar=True)
        return {"loss": test_loss, "preds": preds, "labels": target}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), self.config.train.lr)
        return optimizer

    def save_model(self):
        self.eval()
        tokens_tensor = torch.ones(1, self.config.data.tokanizer_max_len).long()
        mask_tensor = torch.ones(1, self.config.data.tokanizer_max_len).long()
        dummy_input = [(tokens_tensor, mask_tensor, torch.tensor(0, dtype=torch.long))]
        model_scripted = torch.jit.trace(self, dummy_input)  # Export to TorchScript
        model_scripted.save(
            os.path.join(self.config.predict.model_output_dir, self.config.predict.model_name).replace("\\","/")
        )  # Save
        return
