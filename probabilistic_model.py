# Instructions:

# # input: (B, 6) -- corresponds to 6 parameters of the deformation
# # output: (B, 18) -- corresponds to 6x6 stiffness matrix of the material which corresponds to 3 3x3 symmetric submatrices

# # model
# # Take the input, and pass it through a neural network
# # The network should have 6 inputs and 18 outputs. The first 9 are submatrix psi_A, the next 9 are submatrix psi_B, and the last 9 are submatrix psi_D
# # Each psi_i is a 3x3 matrix that is used to define an inverse wishart distribution, but needs to be positive definite using the Cholesky decomposition.
# # We then take the maximum likelihood estimate of the inverse wishart distribution to get the stiffness symmetric 3x3 submatrices A, B, and D, each of which has 6 unique parameters. 
# # This adds up to 18 unique parameters for the 6x6 stiffness matrix and corresponds to the 18 outputs.
# # Then the loss fucntion is the negative log likelihood of the inverse wishart distribution with a gaussian prior on the parameters of the neural network

# PRMOPT:
# Use the instructions in this file to plan a simple training pipeline that loads the data as done in the file @starting_point.ipynb. Give clear sections for loading the data, creating a held out validation step, defining the model, training the model, and validating the trained model.

# ------------------------------------------------------------------------------------------------



import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split

# ── Section 1: Data Loading ─────────────────────────────────────────────────

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)
print(f"Using device: {device}")

x_np = pd.read_csv("data/x_data.csv").to_numpy()
y_np = pd.read_csv("data/y_data.csv").to_numpy()

x_all = torch.tensor(x_np, dtype=torch.float32, device=device)
y_all = torch.tensor(y_np, dtype=torch.float32, device=device)

print(f"Inputs:  {x_all.shape}")
print(f"Targets: {y_all.shape}")

# ── Section 2: Train / Validation Split ──────────────────────────────────────

dataset = TensorDataset(x_all, y_all)
val_fraction = 0.2
val_size = int(len(dataset) * val_fraction)
train_size = len(dataset) - val_size

generator = torch.Generator().manual_seed(42)
train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=len(val_set))

print(f"Train: {train_size} | Val: {val_size}")

# ── Section 3: Model Definition ─────────────────────────────────────────────
#
# y_data column order per submatrix: [11, 22, 33, 23, 13, 12]
# Mapping to 3x3 symmetric matrix indices:
#   index 0 -> (0,0)   index 1 -> (1,1)   index 2 -> (2,2)
#   index 3 -> (1,2)   index 4 -> (0,2)   index 5 -> (0,1)


def _vec6_to_sym33(v):
    """(B, 6) -> (B, 3, 3) symmetric matrix."""
    B = v.shape[0]
    m = torch.zeros(B, 3, 3, device=v.device, dtype=v.dtype)
    m[:, 0, 0] = v[:, 0]
    m[:, 1, 1] = v[:, 1]
    m[:, 2, 2] = v[:, 2]
    m[:, 1, 2] = v[:, 3]
    m[:, 2, 1] = v[:, 3]
    m[:, 0, 2] = v[:, 4]
    m[:, 2, 0] = v[:, 4]
    m[:, 0, 1] = v[:, 5]
    m[:, 1, 0] = v[:, 5]
    return m


def _sym33_to_vec6(m):
    """(B, 3, 3) symmetric matrix -> (B, 6) in [11, 22, 33, 23, 13, 12] order."""
    return torch.stack([
        m[:, 0, 0], m[:, 1, 1], m[:, 2, 2],
        m[:, 1, 2], m[:, 0, 2], m[:, 0, 1],
    ], dim=1)


def _cholesky_params_to_psd(params):
    """(B, 6) raw parameters -> (B, 3, 3) positive-definite matrix via L @ L^T.

    Layout: [diag0, diag1, diag2, L10, L20, L21]
    Diagonal entries pass through softplus to ensure positivity.
    """
    B = params.shape[0]
    L = torch.zeros(B, 3, 3, device=params.device, dtype=params.dtype)
    L[:, 0, 0] = F.softplus(params[:, 0])
    L[:, 1, 1] = F.softplus(params[:, 1])
    L[:, 2, 2] = F.softplus(params[:, 2])
    L[:, 1, 0] = params[:, 3]
    L[:, 2, 0] = params[:, 4]
    L[:, 2, 1] = params[:, 5]
    return L @ L.transpose(-1, -2)


class StiffnessNet(nn.Module):
    def __init__(self, nu: float = 5.0, hidden: int = 128):
        super().__init__()

        # fixed params
        self.P = 3 # sub-matrix dimension
        self.dim_in = 6
        self.dim_out = 18

        self.nu = nu
        self.backbone = nn.Sequential(
            nn.Linear(self.dim_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.dim_out),  # 3 submatrices × 6 Cholesky params
        )

    def forward(self, x):
        raw = self.backbone(x)                       # (B, 18)
        raw_A, raw_B, raw_D = raw.split(6, dim=1)   # 3 × (B, 6)

        psi_A = _cholesky_params_to_psd(raw_A)  # (B, 3, 3)
        psi_B = _cholesky_params_to_psd(raw_B)
        psi_D = _cholesky_params_to_psd(raw_D)

        scale = self.nu + self.P + 1  # nu + 4 for p = 3
        sigma_A = psi_A / scale
        sigma_B = psi_B / scale
        sigma_D = psi_D / scale

        pred = torch.cat([
            _sym33_to_vec6(sigma_A),
            _sym33_to_vec6(sigma_B),
            _sym33_to_vec6(sigma_D),
        ], dim=1)  # (B, 18)

        return pred, (psi_A, psi_B, psi_D)


# ── Section 4: Loss Function ────────────────────────────────────────────────
# TODO: look into this
def inverse_wishart_nll(psi, target_matrix, nu, p=3, eps=1e-4):
    """Inverse-Wishart negative log-likelihood (up to normalising constant).

    Computed in float64 internally to avoid precision loss from the
    wide scale range across submatrices A (~10^1), B (~10^-5), D (~10^3).

    psi:           (B, 3, 3)  predicted scale matrices (positive-definite)
    target_matrix: (B, 3, 3)  observed symmetric matrices
    nu:            scalar, degrees of freedom (must be > p - 1)

    Returns: scalar mean NLL over the batch (back in the input dtype).
    """
    orig_dtype = psi.dtype
    psi64 = psi.to(torch.float64)
    target64 = target_matrix.to(torch.float64) + eps * torch.eye(
        p, device=psi.device, dtype=torch.float64
    )

    _, logdet_psi = torch.linalg.slogdet(psi64)
    _, logdet_x = torch.linalg.slogdet(target64)

    X_inv_Psi = torch.linalg.solve(target64, psi64)
    trace_term = X_inv_Psi.diagonal(dim1=-2, dim2=-1).sum(dim=-1)

    nll = (
        -(nu / 2.0) * logdet_psi
        + ((nu + p + 1) / 2.0) * logdet_x
        + 0.5 * trace_term
    )
    return nll.mean().to(orig_dtype)


def compute_loss(psis, y_batch, nu):
    """Total IW-NLL summed over the three submatrices A, B, D."""
    target_A = _vec6_to_sym33(y_batch[:, 0:6])
    target_B = _vec6_to_sym33(y_batch[:, 6:12])
    target_D = _vec6_to_sym33(y_batch[:, 12:18])

    psi_A, psi_B, psi_D = psis

    return (
        inverse_wishart_nll(psi_A, target_A, nu)
        + inverse_wishart_nll(psi_B, target_B, nu)
        + inverse_wishart_nll(psi_D, target_D, nu)
    )


# ── Section 5: Training ─────────────────────────────────────────────────────

model = StiffnessNet(nu=5.0, hidden=128).to(device)
optimiser = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

n_epochs = 500
print(f"\nModel: StiffnessNet  |  Params: {sum(p.numel() for p in model.parameters())}")
print(f"Training for {n_epochs} epochs...\n")

for epoch in range(n_epochs):
    model.train()
    epoch_loss = 0.0

    for x_batch, y_batch in train_loader:
        pred, psis = model(x_batch)
        loss = compute_loss(psis, y_batch, model.nu)

        optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5) # TODO: look into this
        optimiser.step()

        epoch_loss += loss.item() * x_batch.shape[0]

    epoch_loss /= train_size

    if (epoch + 1) % 25 == 0:
        print(f"  Epoch {epoch+1:>4d}/{n_epochs}  |  Train IW-NLL: {epoch_loss:.4f}")

# ── Section 6: Validation ───────────────────────────────────────────────────

model.eval()
mse_fn = nn.MSELoss()

with torch.no_grad():
    for x_val, y_val in val_loader:
        val_pred, val_psis = model(x_val)

        val_nll = compute_loss(val_psis, y_val, model.nu).item()
        val_mse = mse_fn(val_pred, y_val).item()
        val_mae = (val_pred - y_val).abs().mean().item()

        print(f"\n{'─' * 55}")
        print(f"  Validation IW-NLL : {val_nll:.4f}")
        print(f"  Validation MSE    : {val_mse:.4f}")
        print(f"  Validation MAE    : {val_mae:.4f}")
        print(f"{'─' * 55}")
        print(f"  {'Sub':>3s}  {'MSE':>14s}  {'MAE':>14s}")
        print(f"{'─' * 55}")

        for name, sl in [("A", slice(0, 6)), ("B", slice(6, 12)), ("D", slice(12, 18))]:
            sub_mse = mse_fn(val_pred[:, sl], y_val[:, sl]).item()
            sub_mae = (val_pred[:, sl] - y_val[:, sl]).abs().mean().item()
            print(f"  {name:>3s}  {sub_mse:>14.6f}  {sub_mae:>14.6f}")

        print(f"{'─' * 55}")
