import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
from transformers import Dinov2PreTrainedModel, Dinov2ForImageClassification
from transformers import Dinov2Model, Dinov2PreTrainedModel
from transformers.modeling_outputs import SemanticSegmenterOutput
import random
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


class DinoV2MRIClass(nn.Module):
    def __init__(self, linear_channels=[3], use_hub=True, size='small'):
        super(DinoV2MRIClass, self).__init__()
        self.use_hub = use_hub
        if use_hub:
            model = torch.hub.load('facebookresearch/dinov2', f'dinov2_vit{size[0]}14')
            self.features = model
            in_features = model.num_features
        else:
            model = Dinov2ForImageClassification.from_pretrained(f'facebook/dinov2-{size}-imagenet1k-1-layer')
            self.features = model.dinov2
            in_features = model.dinov2.config.hidden_size


        # Add Global Average Pooling (GAP) layer
        self.gap = nn.AdaptiveAvgPool1d(1)  # Output a single value per feature map

        self.classifier = nn.ModuleList([nn.Dropout(0.2), nn.Linear(in_features, linear_channels[0])])
        if len(linear_channels) > 1:
            for i in range(len(linear_channels)-1):
                self.classifier.append(nn.Linear(linear_channels[i], linear_channels[i+1]))
                if i < len(linear_channels)-2:
                  self.classifier.append(nn.ReLU())

        print(self.classifier)

        for param in self.features.parameters():
            param.requires_grad = False
        for layer in self.classifier:
          if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        # Pass through the feature extractor (EfficientNet blocks)
        x = self.features(x)

        if not self.use_hub:
            x = x.last_hidden_state
            # Apply GAP layer
            x = x.transpose(1, 2)
            x = self.gap(x)
            x = x.squeeze(-1)

        # Final classification layer
        for layer in self.classifier:
          x = layer(x)

        return x


###############################################################################
###############################################################################

class MyEfficientnet_b0(nn.Module):
    def __init__(self, linear_channels):
        super(MyEfficientnet_b0, self).__init__()

        model = models.efficientnet_b0(pretrained=True)
        self.features = model.features
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, linear_channels[0])

        #self.dropout = nn.Dropout(0.2)

        # Add Global Average Pooling (GAP) layer
        self.gap = nn.AdaptiveAvgPool2d(1)  # Output a single value per feature map

        self.classifier = model.classifier

        print(self.classifier)

        for param in self.features.parameters():
            param.requires_grad = False

        for layer in self.classifier:
          if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)

    def forward(self, x):
      x = self.features(x)
      x = self.gap(x)
      x = x.flatten(1)
      x = self.classifier(x)
      return x

######################################################################################################################
######################################################################################################################

class LinearClassifier(torch.nn.Module):
    def __init__(self, in_channels, tokenW=16, tokenH=16, num_labels=1):
        super(LinearClassifier, self).__init__()

        self.in_channels = in_channels
        self.width = tokenW
        self.height = tokenH
        self.classifier = torch.nn.Conv2d(in_channels, num_labels, (1,1))

        nn.init.xavier_uniform_(self.classifier.weight)

    def forward(self, embeddings):
        embeddings = embeddings.reshape(-1, self.height, self.width, self.in_channels)
        embeddings = embeddings.permute(0,3,1,2)

        return self.classifier(embeddings)


class DinoV2MRISeg(Dinov2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.dinov2 = Dinov2Model(config)
        self.classifier = LinearClassifier(config.hidden_size, 16, 16, config.num_labels)

        for name, param in self.named_parameters():
            if name.startswith("dinov2"):
                param.requires_grad = False

    def forward(self, pixel_values, output_hidden_states=False, output_attentions=False, labels=None):
    # use frozen features
        outputs = self.dinov2(pixel_values,
                            output_hidden_states=output_hidden_states,
                            output_attentions=output_attentions)
    # get the patch embeddings - so we exclude the CLS token
        patch_embeddings = outputs.last_hidden_state[:,1:,:]

    # convert to logits and upsample to the size of the pixel values
        logits = self.classifier(patch_embeddings)
        logits = torch.nn.functional.interpolate(logits, size=pixel_values.shape[2:], mode="bilinear", align_corners=False)
        return logits
        # loss = None
        # if labels is not None:
        #   loss_fct = torch.nn.CrossEntropyLoss()
        #   loss = loss_fct(logits.squeeze(), labels.squeeze())
        #
        # return SemanticSegmenterOutput(
        #     loss=loss,
        #     logits=logits,
        #     hidden_states=outputs.hidden_states,
        #     attentions=outputs.attentions,
        # )