from src.boilerplates.models.unet3d        import UNet3D
from src.boilerplates.models.attention_unet import AttentionUNet
from src.boilerplates.models.unetpp        import UNetPP
from src.boilerplates.models.resunet       import ResUNet
from src.boilerplates.models.unetpp_ds     import UNetPPDS
from src.boilerplates.models.swinunetr     import SwinUNETR


def build_model(config):

    model_type = config.model.type

    if model_type == "unet3d":
        return UNet3D(config)

    elif model_type == "attention_unet":
        return AttentionUNet(config)

    elif model_type == "unetpp":
        return UNetPP(config)

    elif model_type == "resunet":
        return ResUNet(config)

    elif model_type == "unetpp_ds":
        return UNetPPDS(config)

    elif model_type == "swinunetr":
        return SwinUNETR(config)

    else:
        raise ValueError(f"Unknown model type: '{model_type}'. "
                         f"Available: unet3d, attention_unet, unetpp, resunet, unetpp_ds, swinunetr")
