import torch
import logging
import numpy as np
from tqdm import tqdm

class EarlyStopping():
    def __init__(self, tolerance=2, min_delta=0.05):
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter +=1
            print(self.counter)
            if self.counter >= self.tolerance:  
                self.early_stop = True

def fit_model(model, trainloader, validloader, epochs, lr, weights_path, device):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    best_valid_loss = np.Inf
    early_stopping = EarlyStopping(tolerance=3, min_delta=0.09)

    train_loss_means, train_acc_means = [], []
    val_loss_means, val_acc_means = [], []

    for epoch in range(1, epochs+1):
        model.train()
        train_loss, train_acc = [], []
        bar = tqdm(trainloader)
        for batch in bar:
            X, y = batch['video'], batch['label']
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = model(X).squeeze(1)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            mean_tl = np.mean(train_loss)

            y_hat = y_hat > 0.5
            co_num = (y == y_hat)
            acc = co_num.sum().item() / len(y)
            # print(y_hat, y, co_num, acc)

            train_acc.append(acc)
            mean_ta = np.mean(train_acc)
            bar.set_description(f"loss {mean_tl:.5f} acc {mean_ta:.5f}")
        train_loss_means.append(mean_tl)
        train_acc_means.append(mean_ta)

        bar = tqdm(validloader)
        val_loss, val_acc = [], []
        model.eval()
        with torch.no_grad():
            for batch in bar:
                X, y = batch['video'], batch['label']
                X, y = X.to(device), y.to(device)
                y_hat = model(X).squeeze(1)
                loss = criterion(y_hat, y)
                val_loss.append(loss.item())
                mean_vl = np.mean(val_loss)
                
                y_hat = y_hat > 0.5
                co_num = (y == y_hat)
                acc = co_num.sum().item() / len(y)
                # print(y_hat, y, co_num, acc)

                val_acc.append(acc)
                mean_va = np.mean(val_acc)
                bar.set_description(f"val_loss {mean_vl:.5f} val_acc {mean_va:.5f}")
            val_loss_means.append(mean_vl)
            val_acc_means.append(mean_va)

        # best valid loss
        if mean_vl < best_valid_loss:
            torch.save(model.state_dict(), weights_path)
            logging.info("WEIGHTS-ARE-SAVED")
            best_valid_loss = mean_vl
        # early stopping
        early_stopping(mean_tl, mean_vl)
        if early_stopping.early_stop:
            logging.info("We are at epoch:", epoch)
            break
        logging.info(f"Epoch {epoch}/{epochs} loss {mean_tl:.5f} val_loss {mean_vl:.5f} acc {mean_ta:.5f} val_acc {mean_va:.5f}")
    return train_loss_means, train_acc_means, val_loss_means, val_acc_means