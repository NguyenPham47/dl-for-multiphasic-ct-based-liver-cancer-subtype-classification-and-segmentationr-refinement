from monai.networks.nets import UNet

def build_model(in_channels=3):
    return UNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=1,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        dropout=0.0,
    )
