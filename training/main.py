import os

import matplotlib.pyplot as plt
import numpy as np
import torch

import models
import opts
from datasets.multidataloader import MultiDataLoader
from epoch import do_epoch
from utils.logger import Logger, savefig
from utils.misc import (
    adjust_learning_rate,
    load_checkpoint_flexible,
    save_checkpoint,
    save_pred,
)
from utils.osutils import mkdir_p


def main(args):
    # seed
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(args.seed)

    args.jointsIx = np.arange(args.pose_dim)

    if not args.debug:
        plt.switch_backend("agg")

    # create checkpoint dir
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # arguments
    opts.print_args(args)
    opts.save_args(args, save_folder=args.checkpoint)

    # create model
    print(
        "==> creating model '{}', in_dim={}, out_dim={}".format(
            args.arch, args.num_in_channels, args.num_classes
        )
    )
    if args.arch == "rgbstream" or args.arch == "flowstream":
        model = models.__dict__[args.arch](
            pretrain_path=None,
            num_classes=args.num_classes,
            num_in_frames=args.num_in_frames,
            inp_res=args.inp_res,
            num_in_channels=args.num_in_channels,
            with_dropout=args.with_dropout,
        )
    elif args.arch == "Pose2Action":
        model = models.Pose2Action(
            num_in_channels=args.num_in_channels,
            num_classes=args.num_classes,
        )
    elif args.arch == "STGCN":
        layout = "smpl"
        if args.pose_rep == "vector_noglobal":
            layout = "smpl_noglobal"
        model = models.STGCN(
            in_channels=args.num_in_channels,
            num_class=args.num_classes,
            graph_args={"layout": layout, "strategy": "spatial"},
            edge_importance_weighting=True,
        )
    else:
        raise ValueError(f"Unrecognized architecture {args.arch}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.nn.DataParallel(model).cuda()
    model = model.to(device)

    # define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss(reduction="mean").cuda()

    if args.optim == "rmsprop":
        optimizer = torch.optim.RMSprop(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    elif args.optim == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    elif args.optim == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

    # optionally resume from a checkpoint
    title = args.datasetname + "-" + args.arch
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            args.start_epoch = checkpoint["epoch"]
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
            logger = Logger(
                os.path.join(args.checkpoint, "log.txt"), title=title, resume=True
            )
            del checkpoint
        else:
            raise ValueError(f"Checkpoint not found at {args.resume}!")
    else:
        logger = Logger(os.path.join(args.checkpoint, "log.txt"), title=title)
        logger_names = ["Epoch", "LR", "train_loss", "val_loss"]
        for p in range(0, args.nloss - 1):
            logger_names.append("train_loss%d" % p)
            logger_names.append("val_loss%d" % p)
        for p in range(args.nperf):
            logger_names.append("train_perf%d" % p)
            logger_names.append("val_perf%d" % p)

        logger.set_names(logger_names)

    if args.pretrained:
        load_checkpoint_flexible(
            model,
            optimizer,
            args,
        )

    param_count = sum(p.numel() for p in model.parameters()) / 1000000.0
    print(f"Total params: {param_count}")

    mdl = MultiDataLoader(args.datasetname)
    train_loader, val_loader, meanstd = mdl._get_loaders(args)

    train_mean = meanstd[0]
    train_std = meanstd[1]
    val_mean = meanstd[2]
    val_std = meanstd[3]

    if args.evaluate or args.evaluate_video:
        print("\nEvaluation only")
        loss, acc, predictions = do_epoch(
            "val",
            val_loader,
            model,
            criterion,
            num_classes=args.num_classes,
            debug=args.debug,
            checkpoint=args.checkpoint,
            mean=val_mean,
            std=val_std,
            num_figs=args.num_figs,
        )
        save_pred(predictions, checkpoint=args.checkpoint)

        # Summarize/save results
        import evaluate

        evaluate.evaluate(val_loader.dataset, exp=args.checkpoint)

        logger_epoch = [0, 0]
        for p in range(len(loss)):  # args.nloss):
            logger_epoch.append(float(loss[p].avg))
            logger_epoch.append(float(loss[p].avg))
        for p in range(len(acc)):  # args.nperf):
            logger_epoch.append(float(acc[p].avg))
            logger_epoch.append(float(acc[p].avg))
        # append logger file
        logger.append(logger_epoch)

        return

    lr = args.lr
    for epoch in range(args.start_epoch, args.epochs):
        lr = adjust_learning_rate(optimizer, epoch, lr, args.schedule, args.gamma)
        print("\nEpoch: %d | LR: %.8f" % (epoch + 1, lr))

        # train for one epoch
        train_loss, train_perf = do_epoch(
            "train",
            train_loader,
            model,
            criterion,
            epochno=epoch,
            optimizer=optimizer,
            num_classes=args.num_classes,
            debug=args.debug,
            checkpoint=args.checkpoint,
            mean=train_mean,
            std=train_std,
            num_figs=args.num_figs,
        )

        # evaluate on validation set
        valid_loss, valid_perf, predictions = do_epoch(
            "val",
            val_loader,
            model,
            criterion,
            epochno=epoch,
            num_classes=args.num_classes,
            debug=args.debug,
            checkpoint=args.checkpoint,
            mean=val_mean,
            std=val_std,
            num_figs=args.num_figs,
        )

        logger_epoch = [epoch + 1, lr]
        for p in range(len(train_loss)):  # args.nloss):
            logger_epoch.append(float(train_loss[p].avg))
            logger_epoch.append(float(valid_loss[p].avg))
        for p in range(len(train_perf)):  # args.nperf):
            logger_epoch.append(float(train_perf[p].avg))
            logger_epoch.append(float(valid_perf[p].avg))
        # append logger file
        logger.append(logger_epoch)

        # Save checkpoint
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "arch": args.arch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            preds=predictions,
            checkpoint=args.checkpoint,
            snapshot=args.snapshot,
        )

        plt.clf()
        plt.subplot(121)
        logger.plot(["train_loss", "val_loss"])
        plt.subplot(122)
        logger.plot(["train_perf0", "val_perf0"])
        savefig(os.path.join(args.checkpoint, "log.eps"))

    logger.close()


if __name__ == "__main__":
    args = opts.parse_opts()
    main(args)
