import os
import shutil
import torch
import scipy.io
from pathlib import Path


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != "numpy":
        raise ValueError("Cannot convert {} to numpy array".format(type(tensor)))
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == "numpy":
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor".format(type(ndarray)))
    return ndarray


def save_checkpoint(
    state, preds, checkpoint="checkpoint", filename="checkpoint.pth.tar", snapshot=1
):
    preds = to_numpy(preds)
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    scipy.io.savemat(os.path.join(checkpoint, "preds.mat"), mdict={"preds": preds})

    if snapshot and state["epoch"] % snapshot == 0:
        shutil.copyfile(
            filepath,
            os.path.join(checkpoint, "checkpoint_%03d.pth.tar" % state["epoch"]),
        )


def load_checkpoint_flexible(
    model,
    optimizer,
    args,
):
    msg = f"no pretrained model found at {args.pretrained}"
    assert Path(args.pretrained).exists(), msg
    print(f"=> loading checkpoint '{args.pretrained}'")
    checkpoint = torch.load(args.pretrained)

    # This part handles ignoring the last layer weights if there is mismatch
    partial_load = False
    if "state_dict" in checkpoint:
        pretrained_dict = checkpoint["state_dict"]
    else:
        print("State_dict key not found, attempting to use the checkpoint:")
        pretrained_dict = checkpoint

    # If the pretrained model is not torch.nn.DataParallel(model), append 'module.' to keys.
    if "module.module" not in sorted(pretrained_dict.keys())[0]:
        # For new resnet50 pretraining
        append_str = "module.module."
        # For old resnet50
        append_str = "module."
        print(f"Appending {append_str} to pretrained keys.")
        pretrained_dict = dict(
            (append_str + k, v) for (k, v) in pretrained_dict.items()
        )

    model_dict = model.state_dict()

    for k, v in pretrained_dict.items():
        if not ((k in model_dict) and v.shape == model_dict[k].shape):
            print(f"Unused from pretrain {k}")
            partial_load = True

    for k, v in model_dict.items():
        if k not in pretrained_dict:
            print(f"Missing in pretrain {k}")
            partial_load = True

    if partial_load:
        print("Removing or not initializing some layers...")
        # 1. filter out unnecessary keys
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if (k in model_dict) and (v.shape == model_dict[k].shape)
        }
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)
        # CAUTION: Optimizer not initialized with the pretrained one
    else:
        print("Loading state dict.")
        model.load_state_dict(checkpoint["state_dict"])
        print("Loading optimizer.")
        optimizer.load_state_dict(checkpoint["optimizer"])

    del checkpoint, pretrained_dict

    if args.freeze_resnet50:
        print("Freezing feature extractor!")
        for layer_cnt, param in enumerate(model.parameters()):
            if layer_cnt >= 159:  # 162:
                print(layer_cnt, param.shape)
            else:
                param.requires_grad = False

    return partial_load


def save_pred(preds, checkpoint="checkpoint", filename="preds_valid.mat"):
    preds = to_numpy(preds)
    filepath = os.path.join(checkpoint, filename)
    scipy.io.savemat(filepath, mdict={"preds": preds})


def adjust_learning_rate(optimizer, epoch, lr, schedule, gamma):
    """Sets the learning rate to the initial LR decayed by schedule"""
    if epoch in schedule:
        lr *= gamma
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
    return lr


# Show num_figs equi-distant images
# If the epoch is too small, show all
def is_show(num_figs, iter_no, epoch_len):
    if num_figs == 0:
        return 0
    show_freq = int(epoch_len / num_figs)
    if show_freq != 0:
        return iter_no % show_freq == 0
    else:
        return 1
