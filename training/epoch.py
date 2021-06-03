import math
import os
import time

import torch

from models import STGCN
from utils import Bar
from utils.evaluation.action import performance, final_preds
from utils.evaluation.averagemeter import AverageMeter
from utils.misc import is_show, save_pred
from utils.vizutils import visualize_gt_pred


# Combined train/val epoch
def do_epoch(
    setname,
    loader,
    model,
    criterion,
    epochno=-1,
    optimizer=None,
    num_classes=None,
    debug=False,
    checkpoint=None,
    mean=torch.Tensor([0.5, 0.5, 0.5]),
    std=torch.Tensor([1.0, 1.0, 1.0]),
    num_figs=100,
):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = [AverageMeter()]
    perfs = [AverageMeter()]

    predictions = torch.Tensor(loader.dataset.__len__(), num_classes)

    if setname == "train":
        model.train()
    elif setname == "val":
        model.eval()

    end = time.time()

    gt_win, pred_win, fig_gt_pred = None, None, None
    bar = Bar("E%d" % (epochno + 1), max=len(loader))
    for i, (inputs, target, meta) in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # Input -> GPU
        input_cuda = inputs.cuda()
        # Target -> GPU
        target_cuda = target.cuda()

        # Forward the input
        if isinstance(model.module, STGCN):
            input_cuda = input_cuda.unsqueeze(4)
        output_cuda = model(input_cuda)

        outputs = output_cuda.data.cpu()
        loss = criterion(output_cuda, target_cuda)
        perfs[0].update(performance(outputs, target), inputs.size(0))

        # measure performance and record loss
        losses[0].update(loss.item(), inputs.size(0))

        # generate predictions
        if setname == "val":
            preds = final_preds(outputs, "dummy", "dummy", "dummy")
            for n in range(outputs.size(0)):
                predictions[meta[0]["index"][n]] = preds[n]

        if debug or is_show(num_figs, i, len(loader)):
            save_path = os.path.join(
                checkpoint, "fig", "pred_%s_epoch%02d_iter%05d" % (setname, epochno, i)
            )
            gt_win, pred_win, fig_gt_pred = visualize_gt_pred(
                inputs,
                outputs,
                target,
                mean,
                std,
                meta[0],
                gt_win,
                pred_win,
                fig_gt_pred,
                save_path=save_path,
                show=debug,
            )

        # compute gradient and do optim step
        if setname == "train":
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = "({batch}/{size}) Data: {data:.1f}s | Batch: {bt:.1f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:} | Perf: {perf:}".format(
            batch=i + 1,
            size=len(loader),
            data=data_time.val,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=", ".join(
                ["{:.3f}".format(losses[i].avg) for i in range(len(losses))]
            ),
            perf=", ".join(["{:.3f}".format(perfs[i].avg) for i in range(len(perfs))]),
        )
        bar.next()

    bar.finish()
    if setname == "train":
        return losses, perfs
    elif setname == "val":
        return losses, perfs, predictions
