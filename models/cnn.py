import torch.nn as nn
import torch.nn.functional as F


class CNN3D(nn.Module):
    """3D CNN for classification of protein data, architecture adapted from
        https://pubs.acs.org/doi/10.1021/acs.jcim.6b00740.

    Args:
            grid_in: Input atom type density grid.
            in_channels (int): Amount of channels in the input grid.
            drop_rate (float, optional): Dropout rate. Defaults to 0.2.
            out_layer (str, optional): Layer type of the output layer.
                Defaults to "sigmoid". Options: "sigmoid", "linear".
            out_classes (int, optional): Number of classification classes.
                Defaults to 1. Defaults to 1.
    """

    def __init__(
        self,
        grid_in,
        in_channels,
        drop_rate=0.2,
        out_layer="sigmoid",
        out_classes=1,
    ):
        super(CNN3D, self).__init__()

        do_rate = drop_rate
        channels = in_channels
        size = grid_in

        self.conv1 = nn.Conv3d(
            in_channels=channels,
            out_channels=32,
            kernel_size=3,
            padding=1,
            stride=1,
        )

        nn.init.xavier_uniform_(self.conv1.weight)
        self.conv2 = nn.Conv3d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1
        )
        nn.init.xavier_uniform_(self.conv2.weight)
        self.conv3 = nn.Conv3d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1
        )
        nn.init.xavier_uniform_(self.conv3.weight)

        self.bn1 = nn.BatchNorm3d(num_features=32)
        self.bn2 = nn.BatchNorm3d(num_features=64)
        self.bn3 = nn.BatchNorm3d(num_features=128)

        last_conv_out = 128
        self.fc1 = nn.Linear(
            in_features=last_conv_out * int(size / 8) * int(size / 8) * int(size / 8),
            out_features=out_classes,
        )
        nn.init.xavier_uniform_(self.fc1.weight)

        if out_layer == "linear":
            self.output = nn.Identity()
        elif out_layer == "sigmoid":
            self.output = nn.Sigmoid()
        elif out_layer == "none":
            self.output = nn.Identity()

        self.dropout = nn.Dropout(p=do_rate)

    def forward(self, x):

        # Max pooling over a (2, 2) window
        x = F.max_pool3d(x, stride=2, kernel_size=2)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)

        x = F.max_pool3d(x, stride=2, kernel_size=2)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn2(x)

        x = F.max_pool3d(x, stride=2, kernel_size=2)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn3(x)

        # flatten
        x = x.view(-1, self.num_flat_features(x))

        x = self.dropout(x)
        x = self.fc1(x)

        # output
        x = self.output(x)

        return x

    @staticmethod
    def num_flat_features(x):
        """Returns the length of the flattened feature vector"""
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
