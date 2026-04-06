from src.boilerplates.models.unet3d        import UNet3D
from src.boilerplates.models.attention_unet import AttentionUNet
from src.boilerplates.models.unetpp        import UNetPP
from src.boilerplates.models.resunet       import ResUNet
from src.boilerplates.models.unetpp_ds     import UNetPPDS


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

    else:
        raise ValueError(f"Unknown model type: {model_type}")
