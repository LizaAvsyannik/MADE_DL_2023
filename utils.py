import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import clear_output
from torchmetrics import CharErrorRate
from tqdm import tqdm


def train_loop(model, loss_func, opt, scheduler, n_epoch, train_loader, val_loader, device, plot_every_epoch=True):
    plt.ion()

    history = []
    val_history = []
    for i in tqdm(range(n_epoch), position=0):
        epoch_loss = []
        for batch_imgs, batch_captions_encoded, _, captions_lengths in tqdm(
            train_loader, position=1, desc='train', leave=False
        ):
            opt.zero_grad()
            labels = batch_captions_encoded.to(device)
            logits = model(batch_imgs.to(device))
            input_lengths = torch.full((batch_imgs.shape[0],), logits.shape[0], dtype=torch.int32)

            loss = loss_func(logits, labels, input_lengths, captions_lengths)

            loss.backward()
            opt.step()

            epoch_loss.append(loss.item())

        history.append(np.mean(epoch_loss))

        with torch.no_grad():
            val_losses = []
            for batch_imgs, batch_captions_encoded, _, captions_lengths in tqdm(
                val_loader, position=2, desc='val', leave=False
            ):
                labels = batch_captions_encoded.to(device)
                logits = model(batch_imgs.to(device))
                input_lengths = torch.full((batch_imgs.shape[0],), logits.shape[0], dtype=torch.int32)

                loss = loss_func(logits, labels, input_lengths, captions_lengths)

                val_losses.append(loss.item())

            val_history.append(np.mean(val_losses))

        scheduler.step(val_history[-1])

        if i + 1 == n_epoch or plot_every_epoch:
            clear_output(wait=True)
            plt.plot(history, label='Train')
            plt.plot(val_history, label='Val')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()

    assert np.mean(history[:10]) > np.mean(history[-10:]), "Model didn't converge."
    print(f'Final Loss: {history[-1]}')


@torch.no_grad()
def eval_model(model, test_loader, device):
    metric = CharErrorRate()
    batch_errors = []
    for batch_imgs, _, batch_captions, _ in test_loader:
        logits = model(batch_imgs.to(device))
        preds = logits.argmax(-1).T.detach().cpu().numpy()
        captions_pred = test_loader.dataset.dataset.decode(preds)
        batch_errors.append(metric(batch_captions, captions_pred))
    return np.mean(batch_errors)


@torch.no_grad()
def analyze_errors(model, test_loader, device):
    metric = CharErrorRate()
    batch_errors = []
    for batch_imgs, _, batch_captions, _ in test_loader:
        logits = model(batch_imgs.to(device))
        preds = logits.argmax(-1).T.detach().cpu().numpy()
        captions_pred = test_loader.dataset.dataset.decode(preds)

        captions_errors = [metric(b_c, c_p) for b_c, c_p in zip(batch_captions, captions_pred)]
        max_erros_indices = np.argwhere(np.array(captions_errors) > 0).flatten()
        batch_imgs_errors = batch_imgs[max_erros_indices]
        batch_captions_errors = [batch_captions[i] for i in max_erros_indices]
        captions_pred_errors = [captions_pred[i] for i in max_erros_indices]

        batch_errors.append((batch_imgs_errors, batch_captions_errors, captions_pred_errors))
    return batch_errors
