import torch
from torch import nn
import torch.nn.functional as F

from models import hourglass, rgbstream

__all__ = ["flowstream"]


def flowstream(**kwargs):
    model = FlowStream(**kwargs)
    model = model.cuda()
    model = nn.DataParallel(model, device_ids=None)
    return model


class FlowStream(nn.Module):
    def __init__(self, **kwargs):
        super(FlowStream, self).__init__()
        self.inp_res = 64
        kwargs["inp_res"] = self.inp_res
        # Flow network
        self.hg = hourglass.hg(num_stacks=2, num_classes=2, num_in_channels=6)
        self.hg = torch.nn.DataParallel(self.hg).cuda()
        # hg_checkpoint = torch.load('pretrained_models/surreal_2_S2_checkpoint.pth.tar')
        hg_checkpoint = torch.load(
            "pretrained_models/flow_surreact_2_v10_fixbg_pretrainsurreal_checkpoint.pth.tar"
        )
        self.hg.load_state_dict(hg_checkpoint["state_dict"])
        del hg_checkpoint
        # Freeze the flow network
        for param in self.hg.parameters():
            param.requires_grad = False

        self.cnn3d = rgbstream(**kwargs)

    def forward(self, x):
        num_batch = x.shape[0]
        num_frames = x.shape[2]
        img_res1 = x.shape[3]
        img_res2 = x.shape[4]
        # flow = torch.zeros(num_batch, 2, num_frames - 1, self.inp_res, self.inp_res)
        flow_list = []
        for t in range(num_frames - 1):
            flow_t = self.hg(
                x[:, :, t : t + 2]
                .permute(0, 2, 1, 3, 4)
                .reshape(num_batch, 6, img_res1, img_res2)
            )[-1]
            if flow_t.shape[-1] != self.inp_res or flow_t.shape[-2] != self.inp_res:
                # Upsample (assumes square input)
                flow_t = F.interpolate(
                    flow_t, scale_factor=self.inp_res / flow_t.shape[-1]
                )
            # flow[:, :, t] = flow_t
            flow_list.append(flow_t)
        flow = torch.stack(flow_list, dim=2)
        # sio.savemat('flow_viz.mat', {'rgb': x.data.cpu().numpy(), 'flow': flow.data.cpu().numpy()})
        x = self.cnn3d(flow)
        return x
