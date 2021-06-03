import mock
import numpy as np
import torch

import datasets


class MultiDataLoader:
    def __init__(self, load_string):
        self.load_string = load_string
        self._set_load_info()

    def _set_load_info(self):
        self.load_dict = {
            "ntu": False,
            "uestc": False,
            "surreact": False,
        }
        sp = self.load_string.split("-")
        for s in sp:
            self.load_dict[s] = True
        print("Data loading info:")
        print(self.load_dict)

    def _get_loaders(self, args):
        train_loader_dict = {}
        val_loader_dict = {}
        # ################## NTU ################### #
        if self.load_dict["ntu"]:
            train_loader_dict["ntu"] = datasets.NTU(
                matfile="data/ntu/info/ntu_data.mat",
                img_folder="data/ntu",
                num_in_frames=args.num_in_frames,
                pose_rep=args.pose_rep,
                inp_res=args.inp_res,
                protocol=args.ntu_protocol,
                views_list_str=args.ntu_views,
                jointsIx=args.jointsIx,
                randframes=args.randframes,
                input_type=args.input_type,
                joints_source=args.joints_source,
            )
            val_loader_dict["ntu"] = datasets.NTU(
                matfile="data/ntu/info/ntu_data.mat",
                img_folder="data/ntu",
                num_in_frames=args.num_in_frames,
                pose_rep=args.pose_rep,
                inp_res=args.inp_res,
                setname=args.test_set,
                evaluate_video=args.evaluate_video,
                protocol=args.ntu_protocol,
                views_list_str=args.ntu_views,
                jointsIx=args.jointsIx,
                randframes=args.randframes,
                input_type=args.input_type,
                joints_source=args.joints_source,
            )
        # ################## UESTC ################### #
        if self.load_dict["uestc"]:
            train_loader_dict["uestc"] = datasets.UESTC(
                img_folder="data/uestc",
                num_in_frames=args.num_in_frames,
                pose_rep=args.pose_rep,
                inp_res=args.inp_res,
                train_views=args.uestc_train_views,
                test_views=args.uestc_test_views,
                jointsIx=args.jointsIx,
                randframes=args.randframes,
            )
            val_loader_dict["uestc"] = datasets.UESTC(
                img_folder="data/uestc",
                num_in_frames=args.num_in_frames,
                pose_rep=args.pose_rep,
                inp_res=args.inp_res,
                setname=args.test_set,
                evaluate_video=args.evaluate_video,
                train_views=args.uestc_train_views,
                test_views=args.uestc_test_views,
                jointsIx=args.jointsIx,
                randframes=args.randframes,
            )
        # ################## SURREACT ################### #
        if self.load_dict["surreact"]:
            train_loader_dict["surreact"] = datasets.SURREACT(
                img_folder="data/surreact/{}".format(args.surreact_version),
                matfile=args.surreact_matfile,
                inp_res=args.inp_res,
                num_in_frames=args.num_in_frames,
                pose_rep=args.pose_rep,
                views_list_str=args.surreact_views,
                jointsIx=args.jointsIx,
                randframes=args.randframes,
                use_segm=args.use_segm,
                use_flow=args.use_flow,
                randbgvid=args.randbgvid,
            )
            val_loader_dict["surreact"] = datasets.SURREACT(
                img_folder="data/surreact/{}".format(args.surreact_version),
                matfile=args.surreact_matfile,
                inp_res=args.inp_res,
                num_in_frames=args.num_in_frames,
                pose_rep=args.pose_rep,
                setname=args.test_set,
                evaluate_video=args.evaluate_video,
                views_list_str=args.surreact_views,
                jointsIx=args.jointsIx,
                randframes=args.randframes,
                use_segm=args.use_segm,
                use_flow=args.use_flow,
                randbgvid=args.randbgvid,
            )
        # ################## -- ################### #
        cnt = 0
        train_loaders = []
        val_loaders = []

        for k in self.load_string.split("-"):
            cnt += 1
            train_loaders += [train_loader_dict[k]]
            val_loaders += [val_loader_dict[k]]

        if cnt == 1:
            dataloader_train = train_loaders[0]
            dataloader_val = val_loaders[0]
            sampler_train = None
            sampler_val = None
            data_shuffle = True
        elif cnt == 2:  # small first, big next
            if len(train_loaders[0]) > len(train_loaders[1]):
                print("Swapping train loaders.")
                train_loaders = [train_loaders[1], train_loaders[0]]
            if len(val_loaders[0]) > len(val_loaders[1]) and not args.watch_first_val:
                print("Swapping val loaders.")
                val_loaders = [val_loaders[1], val_loaders[0]]
            dataloader_train, sampler_train = self.mix_datasets(
                train_loaders[0], train_loaders[1]
            )
            dataloader_val, sampler_val = self.mix_datasets(
                val_loaders[0], val_loaders[1]
            )
            data_shuffle = False
        else:
            print("Up to 2 datasets allowed for now")
            exit()

        # HACK: very quick one:
        if len(dataloader_train) == 0:
            dataloader_train = mock.Mock()
            train_loader = mock.Mock()
            dataloader_train.mean = None
            dataloader_train.std = None
        else:
            # Data loading code
            train_loader = torch.utils.data.DataLoader(
                dataloader_train,
                batch_size=args.train_batch,
                shuffle=data_shuffle,
                num_workers=args.workers,
                pin_memory=True,
                sampler=sampler_train,
            )

        if args.watch_first_val:
            print("Watching first validation")
            print(val_loaders[0])
            val_loader = torch.utils.data.DataLoader(
                # Set this to 0 for ntu-surreact, uestc-surreact
                # No need for this hack anymore since we append with the order in the load_string
                val_loaders[0],
                batch_size=args.test_batch,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=True,
                sampler=None,
            )
        else:
            val_loader = torch.utils.data.DataLoader(
                dataloader_val,
                batch_size=args.test_batch,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=True,
                sampler=sampler_val,
            )

        meanstd = [
            dataloader_train.mean,
            dataloader_train.std,
            dataloader_val.mean,
            dataloader_val.std,
        ]
        return train_loader, val_loader, meanstd

    def mix_datasets(self, data1, data2):
        dataloader = torch.utils.data.ConcatDataset([data1, data2])
        print("Mean/std from first dataloader")
        dataloader.mean = data1.mean
        dataloader.std = data1.std
        len1 = len(data1)
        len2 = len(data2)
        w2 = len1 / (len1 + len2)
        w1 = len2 / (len1 + len2)
        sampler_weights = np.concatenate((w1 * np.ones(len1), w2 * np.ones(len2)))
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sampler_weights, num_samples=2 * len1, replacement=False
        )
        return dataloader, sampler
