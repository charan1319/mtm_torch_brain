from typing import Dict, List
import os
import re
from collections import OrderedDict

import numpy as np
import torch
from einops import repeat

from kirby.data import Data
from kirby.utils import logging

log = logging(header="DATASET", header_color="red")


class Dataset(torch.utils.data.Dataset):
    extension = ".pt"
    pattern = re.compile(r"^(.*)_\d+\.pt$")

    def __init__(
        self, root, split, include=None, transform=None, sequence_len_file=None
    ):
        super().__init__()
        self.root = root

        assert split in ["train", "valid", "test", "finetune"]
        self.split = split

        assert include is not None, "Please specify the datasets to include"
        if isinstance(include, str):
            include = [include]
        self.include = [
            self.__parse_include_arg(include_dir) for include_dir in include
        ]

        self.session_ptr = OrderedDict()
        self.total_num_units = 0

        self.transform = transform
        self.filenames, self.session_ids = self.look_for_files()

        self.session_id_tokens = dict(
            zip(self.session_ptr.keys(), range(len(self.session_ptr)))
        )

        self.sequence_len_file = sequence_len_file

    def __parse_include_arg(self, include):
        session = None
        dataset_dir, session_list = include.split("/")
        if session_list == "*":
            session_list = "all"

        if os.path.exists(os.path.join(self.root, dataset_dir, session_list + ".txt")):
            session_list_filename = session_list + ".txt"
        else:
            session = session_list
            session_list = "all"
        session_list_filename = session_list + ".txt"
        return dataset_dir, session_list_filename, session

    def look_for_files(self):
        files = []
        file_session_ids = []

        for include_dir, include_file_list, only_session in self.include:
            session_ids = self.__parse_file_list(
                os.path.join(self.root, include_dir, include_file_list),
                only_get=only_session,
            )
            include_dir = os.path.join(self.root, include_dir, "processed", self.split)

            for file in sorted(os.listdir(include_dir)):
                if file.endswith(self.extension):
                    session_id = self.parse_session_id(file)
                    if session_id in session_ids:
                        files.append(os.path.join(include_dir, file))
                        file_session_ids.append(session_id)
        return files, file_session_ids

    def register_new_session(self, session_id, num_units):
        if session_id in self.session_ptr:
            raise ValueError(
                f"Session {session_id} already registered, duplicate session files are not allowed."
            )
        self.session_ptr[session_id] = (
            self.total_num_units,
            self.total_num_units + num_units - 1,
        )
        self.total_num_units += num_units

    def __parse_file_list(self, path, only_get=None):
        session_ids = []
        only_get_session_found = False
        with open(path, "r") as f:
            for line in f.readlines():
                session_id, num_units = line.strip().split(" ")
                if only_get is not None:
                    if session_id == only_get:
                        self.register_new_session(session_id, int(num_units))
                        session_ids.append(session_id)
                        only_get_session_found = True
                        break
                else:
                    self.register_new_session(session_id, int(num_units))
                    session_ids.append(session_id)
        if only_get is not None and not only_get_session_found:
            raise ValueError(f"Could not find session {only_get} in file list {path}")
        return session_ids

    def parse_session_id(self, filename):
        filename = os.path.basename(filename)
        match = self.pattern.match(filename)
        if match:
            extracted_session = match.group(1)
            return extracted_session
        else:
            raise ValueError(f"Could not parse session id from filename {filename}")

    def __getitem__(self, item):
        data = torch.load(self.filenames[item])
        # translate unit ids
        session_id = self.session_ids[item]
        translate = self.session_ptr[session_id][0]
        data.spikes.unit_id += translate
        data.units.id += translate
        data.session_id = session_id
        data.session_id_token = self.session_id_tokens[session_id]

        # apply transform
        if self.transform is not None:
            data = self.transform(data)
        return data

    def __len__(self):
        return len(self.filenames)

    def few_shot(self, num_samples, shuffle=True):
        assert num_samples <= len(
            self
        ), f"Cannot sample {num_samples} from dataset of length {len(self)}"
        if shuffle:
            indices = torch.randperm(len(self))
        else:
            indices = torch.arange(len(self))
        self.filenames = [self.filenames[i] for i in indices[:num_samples]]
        self.session_ids = [self.session_ids[i] for i in indices[:num_samples]]
        return self

    def augment_for_batchsize(self, batch_size: int):
        curr_len = len(self)
        if curr_len < batch_size:
            self.filenames = self.filenames * (1 + ((batch_size - 1) // curr_len))
            self.session_ids = self.session_ids * (1 + ((batch_size - 1) // curr_len))
        return self

    def get_sequence_len(self):
        if self.sequence_len_file is None:
            # warn that compute can be slow
            # also if transform is used, this will be wrong
            log.warn(
                "Computing sequence lengths can be slow, consider specifying a sequence length file"
            )
            sequence_len = np.array([len(data.spikes) for data in self])
        else:
            # load npy file
            sequence_len = np.load(self.sequence_len_file)
        return sequence_len


def next_multiple_of_8(x):
    remainder = x % 8
    if remainder == 0:
        return x
    else:
        return x + (8 - remainder)


class Collate:
    def __init__(
        self,
        max_num_units=4096,
        num_latents_per_step=1,
        step=1.0,
        behavior_type_weight=None,
        reweight=False,
        sequence_length=1.0,
    ):
        self.max_num_units = max_num_units + 1
        self.num_latents_per_step = num_latents_per_step
        self.step = step
        self.behavior_type_weight = behavior_type_weight
        self.reweight = reweight
        self.sequence_length = sequence_length

    def __call__(self, batch: List[Data]) -> Dict[str, torch.Tensor | List]:
        # make spike tensors
        num_tokens = [len(data.spikes) + len(data.units.id) * 2 for data in batch]
        max_num_tokens = next_multiple_of_8(max(num_tokens))

        # print(isinstance(batch[0].spikes.timestamps, torch.Tensor))
        # print(batch[0].spikes.timestamps.dtype)
        # print(batch[0].spikes.timestamps.device)
        # print(isinstance(batch[0].spikes.timestamps, np.ndarray))
        # print("---")

        spike_timestamps = torch.zeros(
            (len(batch), max_num_tokens), dtype=torch.float32
        )
        spike_unit = torch.empty((len(batch), max_num_tokens), dtype=torch.long).fill_(
            self.max_num_units - 1
        )
        spike_type = torch.zeros((len(batch), max_num_tokens), dtype=torch.long)
        mask = torch.zeros((len(batch), max_num_tokens), dtype=torch.bool)

        num_output_tokens = [len(data.behavior.timestamps) for data in batch]
        max_num_output_tokens = next_multiple_of_8(max(num_output_tokens))

        # make behavior tensors
        output_timestamps = torch.zeros(
            (len(batch), max_num_output_tokens), dtype=torch.float32
        )
        output_values = torch.empty(
            (len(batch), max_num_output_tokens, 2), dtype=torch.float32
        ).fill_(1e6)
        output_weight = torch.zeros(
            (len(batch), max_num_output_tokens), dtype=torch.float32
        )
        output_stage = torch.zeros(
            (len(batch), max_num_output_tokens), dtype=torch.long
        )

        # make latent tensors
        latent_timestamps = (
            torch.arange(0, self.sequence_length, self.step) + self.step / 2
        )
        latent_ids = torch.arange(self.num_latents_per_step, dtype=torch.long)
        num_timestamps = len(latent_timestamps)
        latent_timestamps = repeat(
            latent_timestamps, "t -> b (t u)", b=len(batch), u=len(latent_ids)
        )
        latent_ids = repeat(latent_ids, "u -> b (t u)", b=len(batch), t=num_timestamps)

        num_timestamps = latent_timestamps.size(1)

        # make attn masks
        input_mask = torch.zeros((len(batch), max_num_tokens), dtype=torch.bool)
        output_mask = torch.zeros((len(batch), max_num_output_tokens), dtype=torch.bool)

        # fill values
        for i, data in enumerate(batch):
            # add spike events
            spikes = data.spikes

            spike_timestamps[i, : len(spikes)] = spikes.timestamps
            spike_unit[i, : len(spikes)] = spikes.unit_id
            mask[i, : len(spikes)] = True
            # add artificial start and end of trial events to each unit
            units = data.units.id
            start, end = data.start, data.end
            # assume that aligned with start and end
            start, end = 0.0, end - start
            spike_timestamps[i, len(spikes) : len(spikes) + len(units)] = start
            spike_timestamps[
                i, len(spikes) + len(units) : len(spikes) + len(units) * 2
            ] = end
            spike_unit[i, len(spikes) : len(spikes) + len(units)] = units
            spike_unit[
                i, len(spikes) + len(units) : len(spikes) + len(units) * 2
            ] = units
            spike_type[i, len(spikes) : len(spikes) + len(units)] = 1
            spike_type[i, len(spikes) + len(units) : len(spikes) + len(units) * 2] = 2

            # make output
            output = data.behavior
            output_timestamps[i, : len(output.timestamps)] = output.timestamps
            output_values[i, : len(output.timestamps)] = output.hand_vel
            output_mask[i, : len(output.timestamps)] = True

            behavior_type = (
                output.type if hasattr(output, "type") else output.behavior_type
            )
            output_stage[i, : len(output.timestamps)] = behavior_type
            output_weight[i, : len(output.timestamps)] = (
                self.behavior_type_weight[behavior_type]
                if self.behavior_type_weight is not None
                else 1.0
            )
            # reweight so that each trial is equally important
            if self.reweight:
                output_weight[i] *= max_num_output_tokens / len(output.timestamps)

            # update masks
            input_mask[i, : len(spikes) + len(units) * 2] = True

        # session id
        session_id = [data.session_id for data in batch]
        task_id = torch.tensor(
            [data.session_id_token for data in batch], dtype=torch.long
        )

        task_id = repeat(task_id, "b -> b t", t=max_num_output_tokens)

        data = dict(
            spike_timestamps=spike_timestamps,
            spike_unit=spike_unit,
            spike_type=spike_type,
            mask=mask,
            output_timestamps=output_timestamps,
            output_values=output_values,
            output_weight=output_weight,
            output_mask=output_mask,
            output_stage=output_stage,
            task_id=torch.clone(task_id),  # repeat and pin_memory don't play well
            latent_timestamps=torch.clone(latent_timestamps),
            latent_id=torch.clone(latent_ids),
            input_mask=input_mask,
            session_id=session_id,
        )
        return data