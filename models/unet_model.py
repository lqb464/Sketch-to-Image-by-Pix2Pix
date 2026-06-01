from .unet_standalone_model import UnetStandaloneModel


class UnetModel(UnetStandaloneModel):
    """Backward-compatible alias for notebooks that pass `--model unet`."""

    pass
