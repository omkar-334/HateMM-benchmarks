import torch.nn as nn

"""
The model consisted of a total of
11 layers. The input vector was given to a convolution layer with 64
filters. This was followed by the next convolution layer of 128 filters.
A max pooling layer with pool size 8 was used for reducing the size
of the vector, followed by a dropout layer with 0.4 as a probability.
The same block of 1D convolution layer, max pooling, and dropout
layers was repeated. This was followed by a flatten layer, a dense
layer, another dropout with 0.4 probability, and the final output dense
layer with activation function softmax. The loss is used for categorical
cross-entropy. The model was trained on the input for 81 epochs. The
resultant prediction confidence scores for each of the audio fragments
of 3s were aggregated and averaged to get the final confidence scores.
The sentiments are then ranked in decreasing order of confidence scores
and picked as predictions based on the number of emotions labeled for
the video.
"""


class AudioNet(nn.Module):
    def __init__(self, input_channels=196, hidden_size=128, num_classes=8):
        super().__init__()

        # Adjust the architecture for 1D input
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=1),  # Changed kernel size to 1
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=1),  # Changed kernel size to 1
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.4),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=1),  # Changed kernel size to 1
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.4),
        )

        # Adjusted flatten size since we removed MaxPool layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, hidden_size),  # Adjusted input size
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(hidden_size, num_classes),
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.fc_layers(x)
        x = self.softmax(x)
        return x


"""The ImageDataGenerator module from
Tensorflow is used to create variations for the given images for a
better input dataset. Once the pre-processing is complete, the data is
given as input in a two-dimensional CNN model (Fukushima & Miyake,
1982). This deep learning model is most commonly used for image
classification, object detection, etc. The model used here is trained with
4 blocks, each consisting of 6 layers including two convolution layers,
two batch normalization layers, a dropout layer, and a max pooling
layer. This is followed by a flattened layer. The next two blocks, each
contain a dense layer, one batch normalization layer, and a dropout
layer. The last layer of the model is a dense layer that gives the final
output. This is also summarized in Fig. 6. The model is run on 133
epochs and gives the result in terms of a vector of probabilities of
occurrence of each of the sentiments."""


class ImageNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=7):  # Also set num_classes=7, as there are 7 output classes
        super().__init__()

        # Create a single block with 6 layers as described
        def create_conv_block(in_channels, out_channels):
            return nn.Sequential(
                # Two convolution layers
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),  # First batch norm
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),  # Second batch norm
                nn.Dropout2d(0.25),  # Dropout layer
                nn.MaxPool2d(kernel_size=2, stride=2),  # Max pooling layer
            )

        # Four blocks of convolutional layers
        self.conv_block1 = create_conv_block(in_channels, 32)
        self.conv_block2 = create_conv_block(32, 64)
        self.conv_block3 = create_conv_block(64, 128)
        self.conv_block4 = create_conv_block(128, 256)

        # Calculate the size of flattened features
        # After 4 blocks with maxpool, size is reduced by 2^4 = 16
        # Input is 48x48, so final feature map is 3x3
        flat_features = 256 * 3 * 3

        # Two dense blocks as described
        self.dense_block1 = nn.Sequential(
            nn.Linear(flat_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        self.dense_block2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        # Final output layer
        self.output_layer = nn.Linear(256, num_classes)

    def forward(self, x):
        # Pass through the four convolutional blocks
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Pass through dense blocks
        x = self.dense_block1(x)
        x = self.dense_block2(x)

        # Final output
        x = self.output_layer(x)
        return x
