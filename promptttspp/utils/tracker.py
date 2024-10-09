# Copyright 2024 LY Corporation

# LY Corporation licenses this file to you under the Apache License,
# version 2.0 (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at:

#   https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

from collections import defaultdict


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count

    def mean(self):
        return self.avg


class LossTracker(defaultdict):
    def __init__(self):
        super().__init__(AverageMeter)

    def update(self, **kwargs):
        for k, v in kwargs.items():
            self[k].update(v)

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)


class Tracker:
    def __init__(self, log_file, mode="w"):
        super().__init__()
        self.loss_tracker = LossTracker()
        self.log_file = log_file
        self.mode = mode

    def update(self, **kwargs):
        self.loss_tracker.update(**kwargs)

    def update_with_prefix(self, prefix, **kwargs):
        self.loss_tracker.update(**{prefix + k: v for k, v in kwargs.items()})

    def items(self):
        return self.loss_tracker.items()

    def __getattr__(self, key):
        return self.loss_tracker.get(key)

    def __getitem__(self, key):
        return self.loss_tracker.get(key)

    def write(self, epoch, clear=True):
        if epoch == 1:
            with open(self.log_file, self.mode) as f:
                f.write(",".join(["epoch"] + list(self.loss_tracker.keys())) + "\n")
                f.write(
                    ",".join(
                        [str(epoch)]
                        + [str(v.mean()) for v in self.loss_tracker.values()]
                    )
                    + "\n"
                )
        else:
            with open(self.log_file, "a") as f:
                f.write(
                    ",".join(
                        [str(epoch)]
                        + [str(v.mean()) for v in self.loss_tracker.values()]
                    )
                    + "\n"
                )

        if clear:
            self.loss_tracker.clear()
