from AnomalyDetectionVit.utils.ckpt_util import load_ckpt





def load_unet_stageA(model, ckpt_path, device):
    ckpt = load_ckpt(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["unet"], strict=True)

    return model.to(device)