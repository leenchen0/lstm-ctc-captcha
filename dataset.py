# coding: utf-8

import os

from PIL import Image
import numpy as np

from captcha_generator import CHARSET

encoded_map = {}
decode_map = {}
for i, c in enumerate(CHARSET):
    encoded_map[c] = i
    decode_map[i] = c

class dataset(object):

    def __init__(self, folder, batch_size):
        self.folder = folder
        self.batch_size = batch_size

        self._data = []
        self._labels = []
        self._size = 0
        self._pos = 0

        self._load()

    def _load(self):
        """Load all .png file in self.folder"""
        data = []
        labels = []
        for root, _, files in os.walk(self.folder):
            for filename in files:
                if filename.endswith('.png'):
                    img = Image.open(os.path.join(root, filename))
                    data.append(np.array(img) / 255)
                    labels.append([encoded_map[c] for c in filename[:filename.rfind('.')]])

        self._size = len(data)
        self._data = data
        self._labels = labels

        self._shuffle()

    def _shuffle(self):
        z = list(zip(self._data, self._labels))
        np.random.shuffle(z)
        self._data, self._labels = zip(*z)

    @property
    def data(self):
        return self._data

    @property
    def labels(self):
        return self._labels

    def next_batch(self):
        """
        Returns:
            batch: list
                len(batch) = self.batch_size
            labels: list
            new_epoch: Boolean
                Indicates whether a loop is completed
        """

        batch = []
        labels = []

        count = 0
        new_epoch = False
        while count < self.batch_size:
            need = self.batch_size - count

            cur_pos = min(self._size, self._pos + need)
            count += cur_pos - self._pos
            batch.extend(self._data[self._pos:cur_pos])
            labels.extend(self._labels[self._pos:cur_pos])

            self._pos = cur_pos if cur_pos < self._size else 0
            if cur_pos == self._size:
                new_epoch = True
                self._shuffle()

        return batch, labels, new_epoch
