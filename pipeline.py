import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sympy import evaluate
import torch
import torch.nn as nn
from  torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F 

from torchgen import model
import wfdb
from wfdb import processing
import torchmetrics as tm
from torchmetrics.aggregation import MeanMetric
from sklearn.model_selection import train_test_split

import warnings
import tqdm


from ODConv1d import ECGUNet
# ════════════════════════════════════════════════════════════════════════════
#  Dataset
# ════════════════════════════════════════════════════════════════════════════

class ECGRpeakDataset(Dataset):
    """
    PyTorch Dataset for R-peak detection.

    Args:
        input:      list of 1-D numpy arrays [L].
        target: list of R-peak index arrays.
        sigma:        Gaussian width in samples.
        length:       Signal length L (all signals must share the same L).
        normalize:    z-score normalise each signal independently.

    Returns per sample:
        x : [1, L]  float32  — normalised ECG signal
        y : [1, L]  float32  — Gaussian heatmap target in (0, 1]
    """

    def __init__(  self,  input:  list, target: list,  sigma: float = 7.0, length: int = 3600,
        normalize:    bool  = True,  ) -> None:
        
        assert len(input) == len(target), \
            "input and target must have the same length."
        self.input      = input
        self.target = target
        self.normalize    = normalize
        #self.builder      = GaussianTargetBuilder(sigma=sigma, length=length)

    def __len__(self) -> int:
        return len(self.input)

    def __getitem__(self, idx: int):
        signal = np.asarray(self.input[idx], dtype=np.float32)
        # if self.normalize:
        #     std = signal.std()
        #     if std > 1e-8:
        #         signal = (signal - signal.mean()) / std
        x = torch.from_numpy(signal)#.unsqueeze(0)                    # [1, L]
        y = torch.from_numpy(self.target[idx])#.unsqueeze(0)          # [1, L]
        return x, y


# ════════════════════════════════════════════════════════════════════════════
#  Loss function
# ════════════════════════════════════════════════════════════════════════════

class CombinedLoss(nn.Module):
    """
    MSE + BCE combined loss for Gaussian heatmap regression.

    MSE  on sigmoid(logits) vs target  — shapes Gaussian profile accurately.
    BCE  on raw logits      vs target  — handles class imbalance
                                         (background >> R-peak samples).

    Args:
        mse_weight: weight multiplier for MSE term (default 1.0).
        bce_weight: weight multiplier for BCE term (default 1.0).

    Inputs:
        pred_logits: [B, 1, L]  raw model output (before sigmoid)
        target:      [B, 1, L]  Gaussian heatmap values in (0, 1)
    """

    def __init__(self, mse_weight: float = 1.0, bce_weight: float = 1.0) -> None:
        super().__init__()
        self.mse_weight = mse_weight
        self.bce_weight = bce_weight

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_prob = torch.sigmoid(pred_logits)
        loss_mse  = F.mse_loss(pred_prob, target)
        loss_bce  = F.binary_cross_entropy_with_logits(pred_logits, target)
        return self.mse_weight * loss_mse + self.bce_weight * loss_bce



class Training:
    
    def __init__(self, model, train_loader, val_loader, test_loader, loss_fn, optimizer, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.metric = MeanMetric().to(device)
    
    
    
    def train_one_epoch(self, epoch=None):
        self.model.train()
        loss_train = MeanMetric()
        self.metric.reset()

        with tqdm.tqdm(self.train_loader, unit='batch') as tepoch:
          for inputs, targets in tepoch:
            if epoch:
              tepoch.set_description(f'Epoch {epoch}')
            
            
            inputs = inputs.unsqueeze(1).float().to(self.device)
            targets = targets.unsqueeze(1).float().to(self.device)

            outputs = self.model(inputs)

            loss = self.loss_fn(outputs, targets)

            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            loss_train.update(loss.item(), weight=len(targets))
            self.metric.update(outputs, targets)

            tepoch.set_postfix(loss=loss_train.compute().item(),
                               metric=self.metric.compute().item())

        return self.model, loss_train.compute().item(), self.metric.compute().item()

    
    def evaluate(self, model, test_loader, loss_fn, metric):
        model.eval()
        loss_eval = MeanMetric().to(self.device)
        metric.reset()
    
        with torch.inference_mode():
          for inputs, targets in test_loader:
            inputs = inputs.unsqueeze(1).float().to(self.device)
            targets = targets.unsqueeze(1).float().to(self.device)

            outputs = model(inputs)
    
            loss = loss_fn(outputs, targets)
            loss_eval.update(loss.item(), weight=len(targets))
    
            metric(outputs, targets)
    
        return loss_eval.compute().item(), metric.compute().item()
    
    def train(self,  num_epochs):
        loss_train_hist = []
        loss_valid_hist = []

        metric_train_hist = []
        metric_valid_hist = []

        best_loss_valid = torch.inf
        epoch_counter = 0
        
        for epoch in range(num_epochs):
            # Train
            model, loss_train, metric_train = self.train_one_epoch( epoch)
            # Validation
            loss_valid, metric_valid = self.evaluate(model,
                                      self.val_loader,
                                      self.loss_fn,
                                      self.metric)

            loss_train_hist.append(loss_train)
            loss_valid_hist.append(loss_valid)

            metric_train_hist.append(metric_train)
            metric_valid_hist.append(metric_valid)

            if loss_valid < best_loss_valid:
                torch.save(model, f'model.pt')
                best_loss_valid = loss_valid
                print('Model Saved!')

        plt.plot(loss_train_hist, label='train')
        plt.plot(loss_valid_hist, label='validation')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()
    
    
    






if __name__ == "__main__":
    if torch.cuda.is_available():
        print("CUDA is available. Using GPU.")

 
    
    
    X = np.load("dataset/input.npy")
    Y = np.load("dataset/target.npy") 

    x_train, x_temp, y_train, y_temp = train_test_split( X, Y, test_size=0.20, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.50, random_state=42)

    print(len(x_train), len(x_val), len(x_test))
    
    trainset = ECGRpeakDataset( x_train, y_train)
    valset = ECGRpeakDataset( x_val, y_val)
    testset = ECGRpeakDataset( x_test, y_test)

    train_loader = DataLoader(trainset, batch_size=32, shuffle=True)
    val_loader = DataLoader(valset, batch_size=32, shuffle=True)
    test_loader = DataLoader(testset, batch_size=1, shuffle=True)

    
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # loss_fn = CombinedLoss(mse_weight=1.0, bce_weight=1.0).to(device)
    
    # model = ECGUNet(in_channels=1, out_channels=1, kernel_size=7, kernel_num= 4, reduction= 0.0625).to(device)
    # optimizer    = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    
    # trainer= Training(model, train_loader, val_loader, test_loader, loss_fn, optimizer, device)
    # trainer.train(num_epochs=10)


    # ════════════════════════════════════════════════════════════════════════════
    #  Inference — heatmap → R-peak sample indices
    # ════════════════════════════════════════════════════════════════════════════

def predict_rpeaks(
    model:     ECGUNet,
    x:         torch.Tensor,
    threshold: float = 0.4,
    min_dist:  int   = 72,
    device:    str   = "cpu",
    ) ->     list[np.ndarray]:
    """
    Convert model heatmap output to R-peak sample indices.
    Steps:
        1. model(x)   → raw logits [B, 1, L]
        2. sigmoid     → probability map in (0, 1)
        3. threshold   → candidate positions with prob > threshold
        4. NMS         → greedy suppression within min_dist window
    Args:
        model:     trained ECGUNet.
        x:         ECG tensor [B, 1, L].
        threshold: minimum probability to consider a candidate (default 0.5).
        min_dist:  minimum samples between two R-peaks.
               Rule of thumb: sampling_rate * 0.2  (200 ms refractory)
                 360 Hz → 72,   500 Hz → 100
    device:    'cuda' or 'cpu'.
    Returns:
        List[np.ndarray] of length B — sorted R-peak indices per sample.
    """
    model.eval()
    with torch.no_grad():
        heatmap = torch.sigmoid(model(x.to(device)))  # [B, 1, L]
    heatmap = heatmap.squeeze(1).cpu().numpy()         # [B, L]
    results = []
    for prob in heatmap:
        candidates = np.where(prob > threshold)[0]
        if len(candidates) == 0:
            results.append(np.array([], dtype=np.int64))
            continue
        # Greedy NMS: pick highest peak first, suppress window around it
        peaks      = []
        suppressed = np.zeros(len(prob), dtype=bool)
        for idx in candidates[np.argsort(prob[candidates])[::-1]]:
            if suppressed[idx]:
                continue
            peaks.append(idx)
            lo = max(0, idx - min_dist)
            hi = min(len(prob), idx + min_dist + 1)
            suppressed[lo:hi] = True
        results.append(np.sort(np.array(peaks, dtype=np.int64)))
    return results




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loaded_model = torch.load(
    "model.pt",
    map_location=device,
    weights_only=False
)
loaded_model.to(device)
loaded_model.eval()
e= iter(test_loader)
input, targets = next(e)
print(input.shape, targets.shape)

rpeaks= predict_rpeaks(loaded_model, input.unsqueeze(1).to("cuda"), threshold=0.4, min_dist=72, device="cuda")



time = np.arange(len(input[0])) 
normalized_input = (input[0] - input[0].mean()) / input[0].std()
plt.figure(figsize=(14, 4))
plt.plot(time, normalized_input, label="ECG")
plt.scatter(
    time[rpeaks[0]],
    rpeaks[0],
    color="red",
    label="Predicted R-peaks"
)
# plt.figure(figsize=(12, 6))
# plt.subplot(2, 1, 1)
# plt.scatter(rpeaks[0], [1] * len(rpeaks[0]), c='red', s=50, label='Predicted R-peaks')
# plt.subplot(2, 1, 2)
# plt.plot(input[0])
plt.show()
