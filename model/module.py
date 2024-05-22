import torch.nn as nn


class BasicConv2d(nn.Module):
    def __init__(self, in_features, out_features, kernel_size: int = 3, stride: int = 1, dilation: int = 1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.layers = nn.Sequential(
            nn.Conv2d(in_features, out_features,
                      kernel_size=(kernel_size, kernel_size),
                      stride=(stride, stride),
                      padding=padding,
                      dilation=(dilation, dilation)),
            nn.BatchNorm2d(out_features),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layers(x)


class ImageModule(nn.Module):
    def __init__(self, input_shape=(3, 400, 650), out_features=64, classify_sum: int = 4):
        super(ImageModule, self).__init__()
        self.ms_cam = MS_CAM(channels=64) # MS_CAM 模块
        self.layers = nn.Sequential(
            BasicConv2d(input_shape[0], 8),
            BasicConv2d(8, 8, stride=2),
            BasicConv2d(8, 8, stride=2),
            BasicConv2d(8, 16, stride=2),
            BasicConv2d(16, 16, stride=2),
            BasicConv2d(16, 32, stride=2),
            BasicConv2d(32, 32, stride=2),
            BasicConv2d(32, 64, stride=2),
            BasicConv2d(64, out_features, stride=2)
        )
        conv2d_count = 8
        H = int(input_shape[1] // (2 ** conv2d_count)) + 1
        W = int(input_shape[2] // (2 ** conv2d_count)) + 1
        in_features = out_features * H * W
        
        self.regression = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, int(in_features / 2)),
            nn.ReLU(),
            nn.Linear(int(in_features / 2), int(in_features / 4)),
            nn.ReLU(),
            nn.Linear(int(in_features / 4), 1)
        )

        self.classify = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, int(in_features / 2)),
            nn.ReLU(),
            nn.Linear(int(in_features / 2), int(in_features / 4)),
            nn.ReLU(),
            nn.Linear(int(in_features / 4), classify_sum),
            # nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layers(x) # x has 64 channels
        x = self.ms_cam(x)
        y_regression = self.regression(x)
        y_classify = self.classify(x)
        return y_regression, y_classify


class CSVModule(nn.Module):
    def __init__(self, in_features: int, classify_sum: int = 2):
        super(CSVModule, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
        )

        self.regression = nn.Sequential(
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 1)
        )

        self.classify = nn.Sequential(
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, classify_sum),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layers(x)
        y_regression = self.regression(x)
        y_classify = self.classify(x)
        return y_regression, y_classify


class MS_CAM(nn.Module):
    def __init__(self, channels=64, r=4):
        super(MS_CAM, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        return x * wei
