import os

import torch
from PIL import Image
from torch import nn
from torchvision import transforms


class CaptionsDataset(torch.utils.data.Dataset):
    _BLANK_TOKEN = '0'

    def __init__(self, image_folder, transform=None):
        self._image_folder = image_folder
        self._image_paths = os.listdir(image_folder)
        self._captions = [''.join(path.split('.')[:-1]) for path in self._image_paths]
        self._init_token_maps()

        self._image_transform = transform
        self._to_tensor_tf = transforms.ToTensor()
        self._images = {path: self._read_image(path) for path in self._image_paths}
        self._encoded_captions = self.encode(self._captions)
        
    def _init_token_maps(self):
        tokens = set(''.join(self._captions))
        self._tokens = sorted(tokens)
        self._idx_to_token = {0: self._BLANK_TOKEN}
        self._idx_to_token.update({i + 1: c for i, c in enumerate(self._tokens)})
        self._token_to_idx = {self._BLANK_TOKEN: 0}
        self._token_to_idx.update({c: i + 1 for i, c in enumerate(self._tokens)})

    def _read_image(self, path):
        image_path = os.path.join(self._image_folder, path)
        x = Image.open(image_path)
        x = self._to_tensor_tf(x)

        if x.shape[0] == 4:
            assert torch.allclose(x[3], torch.ones([x.shape[1], x.shape[2]]))
            x = x[:3]
        return x

    def __getitem__(self, index):
        x = self._images[self._image_paths[index]]
        if self._image_transform is not None:
            x = self._image_transform(x)
        encoded_caption = self._encoded_captions[index]
        caption = self._captions[index]
        return (x, encoded_caption, caption, encoded_caption.shape[0])
    
    def __len__(self):
        return len(self._image_paths)

    @property
    def idx_to_token(self):
        return self._idx_to_token
    
    @property
    def token_to_idx(self):
        return self._token_to_idx

    @property
    def num_tokens(self):
        return len(self._tokens) + 1

    @property
    def blank_token(self):
        return self._BLANK_TOKEN

    def encode(self, sequences, max_len=None):    
        max_len = max_len or max(map(len, sequences))
        sequences_idx = []

        for sequence in sequences:
            line_idx = torch.LongTensor([self._token_to_idx[c] for c in sequence])
            sequences_idx.append(line_idx)

        return nn.utils.rnn.pad_sequence(
            sequences=sequences_idx,
            batch_first=True,
            padding_value=0,
        ).long()

    def decode(self, index_sequences):
        def _construct_caption(idx_sequence):
            token_sequence = []
            last_idx = -1
            for idx in idx_sequence:
                token = self._idx_to_token[idx]
                if token == self.blank_token:
                    last_idx = -1
                    continue
                elif idx == last_idx:
                    continue
                token_sequence.append(token)
                last_idx = idx
            return ''.join(token_sequence)

        return list(map(_construct_caption, index_sequences))
