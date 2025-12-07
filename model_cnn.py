import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class SimpleCNN(nn.Module):
    def __init__(self, layer_config, num_classes=3, input_size=(3, 224, 224)):
        super(SimpleCNN, self).__init__()
        self.input_size = input_size
        in_classifier = False
        self.feature_config = []
        self.classifier_config = []
        for layer in layer_config:
            if layer["type"] == "linear":
                in_classifier = True
                self.classifier_config.append(layer)
            elif in_classifier:
                self.classifier_config.append(layer)
            else:
                self.feature_config.append(layer)
        self.features = self._make_layers(self.feature_config)
        self.flatten_dim = self._get_flatten_size()
        self.classifier = self._make_classifier(self.classifier_config, self.flatten_dim, num_classes)
    def _make_layers(self, config):
        layers = []
        in_channels = self.input_size[0]
        for layer in config:
            if layer["type"] == "conv":
                layers.append(nn.Conv2d(in_channels, layer["out_channels"], layer["kernel_size"]))
                in_channels = layer["out_channels"]
            elif layer["type"] == "batchnorm":
                layers.append(nn.BatchNorm2d(in_channels))
            elif layer["type"] == "pool":
                layers.append(nn.MaxPool2d(kernel_size=layer["kernel_size"], stride=layer.get("stride", 2)))
            elif layer["type"] == "relu":
                layers.append(nn.ReLU())
        return nn.Sequential(*layers)
    def _make_classifier(self, config, flatten_dim, num_classes):
        layers = []
        in_features = flatten_dim
        for layer in config:
            if layer["type"] == "linear":
                out_features = layer["out_features"]
                layers.append(nn.Linear(in_features, out_features))
                in_features = out_features
            elif layer["type"] == "dropout":
                layers.append(nn.Dropout(p=layer.get("rate", 0.5)))
            elif layer["type"] == "relu":
                layers.append(nn.ReLU())
        if not any(l["type"] == "linear" and l.get("out_features") == num_classes for l in config):
            layers.append(nn.Linear(in_features, num_classes))
        return nn.Sequential(*layers)
    def _get_flatten_size(self):
        with torch.no_grad():
            dummy = torch.zeros(1, *self.input_size)
            out = self.features(dummy)
            return out.view(1, -1).size(1)
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    def print(self, verbose=True):
        print("Model Architecture:\n")
        print(self)
        input_size = self.input_size
        try:
            summary(self, input_size)
        except ImportError:
            print("\nInstall `torchsummary` for a nicer output:\n  pip install torchsummary")
