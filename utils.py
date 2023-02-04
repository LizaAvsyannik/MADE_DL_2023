import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import clear_output
from PIL import Image
from torchvision import transforms


class CaptionsDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.image_paths = os.listdir(image_folder)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_folder, self.image_paths[index])
        x = Image.open(image_path)
        to_tensor = transforms.ToTensor()
        x = to_tensor(x)

        if x.shape[0] == 4:
            assert torch.allclose(x[3], torch.ones([x.shape[1], x.shape[2]]))
            x = x[:3]
    
        if self.transform is not None:
            x = self.transform(x)
        
        caption = self.image_paths[index][:-4]
        return (x, caption)
    
    def __len__(self):
        return len(self.image_paths)


def to_matrix(token_to_idx, sequences, max_len=None, dtype='int32'):    
    max_len = max_len or max(map(len, sequences))
    sequences_ix = np.zeros([len(sequences), max_len], dtype)

    for i in range(len(sequences)):
        line_ix = [token_to_idx[c] for c in sequences[i]]
        sequences_ix[i, :len(sequences[i])] = line_ix

    return sequences_ix


def to_caption(idx_to_token, sequences, blank_idx=0):    
    captions = []

    for i in range(len(sequences)):
        caption_ix = ''.join([idx_to_token[c] for c in sequences[i] if c != blank_idx])
        captions.append(caption_ix)

    return captions


def train_loop(model, loss_func, opt, n_epoch, token_to_idx, train_loader, device):
    history = []
    for i in range(n_epoch):
        epoch_loss = []
        for batch_imgs, batch_captions in train_loader:
            labels = torch.tensor(to_matrix(token_to_idx, batch_captions), dtype=torch.int32).to(device)
            logits = model(batch_imgs.to(device)).permute(1, 0, 2)
            input_lengths = torch.full((batch_imgs.shape[0],), logits.shape[0], dtype=torch.int32)
            target_lengths = torch.full((batch_imgs.shape[0],), labels.shape[1], dtype=torch.int32)

            # print(labels.shape, logits.shape, input_lengths[0], target_lengths[0])
            loss = loss_func(logits.log_softmax(-1), labels, input_lengths, target_lengths)

            loss.backward()
            opt.step()
            opt.zero_grad()
            
            epoch_loss.append(loss.item())

        history.append(sum(epoch_loss) / len(epoch_loss))
        if (i + 1) % 10 == 0:
            clear_output(True)
            plt.plot(history, label='Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()

    assert np.mean(history[:10]) > np.mean(history[-10:]), "Model didn't converge."
    print(f'Final Loss: {history[-1]}')


