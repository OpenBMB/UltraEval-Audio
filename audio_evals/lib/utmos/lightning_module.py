import pytorch_lightning as pl
import torch
import torch.nn as nn
from model import load_ssl_model, DomainEmbedding, LDConditioner, Projection
import os


class BaselineLightningModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.construct_model()
        self.save_hyperparameters()

    def construct_model(self, path=None):
        if path is None:
            assert (
                os.environ.get("SSL_MODEL_PATH") is not None
            ), "SSL_MODEL_PATH is not set"
            path = os.environ.get("SSL_MODEL_PATH")

        self.feature_extractors = nn.ModuleList(
            [
                load_ssl_model(cp_path=path),
                DomainEmbedding(3, 128),
            ]
        )
        output_dim = sum(
            [
                feature_extractor.get_output_dim()
                for feature_extractor in self.feature_extractors
            ]
        )
        output_layers = [
            LDConditioner(judge_dim=128, num_judges=3000, input_dim=output_dim)
        ]
        output_dim = output_layers[-1].get_output_dim()
        output_layers.append(
            Projection(
                hidden_dim=2048,
                activation=torch.nn.ReLU(),
                range_clipping=False,
                input_dim=output_dim,
            )
        )

        self.output_layers = nn.ModuleList(output_layers)

    def forward(self, inputs):
        outputs = {}
        for feature_extractor in self.feature_extractors:
            outputs.update(feature_extractor(inputs))
        x = outputs
        for output_layer in self.output_layers:
            x = output_layer(x, inputs)
        return x
