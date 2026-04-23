import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

from mha import MultiHeadAttention

# def x_to_seq(x, T):
#     """
#     Convert a batch of 2D points x ∈ (N, 2) into sinusoidal sequences of length T.

#     Returns: (N, T, 2) tensor
#     """
#     N, d = x.shape
#     assert d == 2

#     t = torch.linspace(0, 1, T, device=x.device).unsqueeze(0).unsqueeze(-1)  # (1, T, 1)
#     x_expanded = x.unsqueeze(1)  # (N, 1, 2)

#     seq = torch.sin(2 * torch.pi * (x_expanded * t))  # (N, T, 2)
#     return seq


# --------------------------------------------------
# x_to_seq function
# --------------------------------------------------
def x_to_seq(x: torch.Tensor, T: int = 20) -> torch.Tensor:
    """Map x of shape (B, 2) to a length-T sequence of 2D tokens.

    Here x is interpreted as parameters (frequency, phase) of a sinusoid,
    and the sequence consists of sampled points on the corresponding
    2D curve:

        token_t = [sin(omega * tau_t + phi), cos(omega * tau_t + phi)].

    This gives a genuinely sequence-like input where attention can
    exploit structure across time.
    """
    B, d_in = x.shape
    assert d_in == 2, "x_to_seq assumes d_in = 2"
    device = x.device

    # Map x to frequency and phase via a squashing nonlinearity
    omega = 2.0 * torch.pi * torch.sigmoid(x[:, 0])  # (B,)
    phi   = 2.0 * torch.pi * torch.sigmoid(x[:, 1])  # (B,)

    # Discrete time grid in [0, 1]
    tau = torch.linspace(0.0, 1.0, T, device=device)  # (T,)

    tokens = []
    for t in range(T):
        tau_t = tau[t]
        # Broadcast omega, phi over batch
        angle = omega * tau_t + phi          # (B,)
        t_tok = torch.stack(
            [
                torch.sin(angle),
                torch.cos(angle),
            ],
            dim=-1,
        )  # (B, 2)
        tokens.append(t_tok)

    seq = torch.stack(tokens, dim=1)  # (B, T, 2)
    return seq


# --------------------------------------------------
# Utility: classification plot x ∈ R^2 ↦ argmax(model(x))
# --------------------------------------------------
def plot_classification(
    model,
    device,
    x_min=-2.0,
    x_max=2.0,
    grid_size=200,
    title="decision",
    seq_len=10,
):
    """
    Plot the classification regions induced by `model` on a 2D grid in [x_min, x_max]^2.

    model: MultiHeadAttention with d_in = 2, d_out = C (here C=2)
    """
    model.eval()

    xs = torch.linspace(x_min, x_max, grid_size)
    ys = torch.linspace(x_min, x_max, grid_size)
    Xg, Yg = torch.meshgrid(xs, ys, indexing="xy")  # (G, G)

    # Grid points in R^2
    grid = torch.stack([Xg, Yg], dim=-1).reshape(-1, 2).to(device)  # (G^2, 2)

    with torch.no_grad():
        # Map grid points to sinusoidal sequences: (N, seq_len, 2)
        seq = x_to_seq(grid, seq_len)
        logits_all = model(seq)                    # (N, seq_len, d_out)
        # Aggregate logits over the token dimension (mean pooling)
        logits = logits_all.mean(dim=1)            # (N, d_out)
        labels = torch.argmax(logits, dim=-1)      # (N,)

    labels = labels.view(grid_size, grid_size).cpu().numpy()

    plt.figure(figsize=(4, 4))
    plt.imshow(
        labels.T,
        origin="lower",
        extent=(x_min, x_max, x_min, x_max),
        aspect="equal",
        interpolation="nearest",
    )
    plt.colorbar(label="class")
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.tight_layout()
    plt.show()


# --------------------------------------------------
# Data generation from teacher MHA
# --------------------------------------------------
def generate_data(
    teacher_mha,
    n_samples,
    seq_len,
    noise=0.0,
    device="cpu",
    x_min=-2.0,
    x_max=2.0,
):
    """
    Generate (x, label) pairs using a teacher MHA.

    - x ∈ R^{n_samples × 2}, uniformly in [x_min, x_max]^2
    - label = argmax over teacher_mha(x) logits
    - noise (float): standard deviation of Gaussian noise added to logits
    """
    teacher_mha.eval()

    # Uniform samples in a finite square range
    x = torch.empty(n_samples, 2).uniform_(x_min, x_max).to(device)

    with torch.no_grad():
        seq = x_to_seq(x, seq_len)           # (N, seq_len, 2)
        logits_all = teacher_mha(seq)        # (N, seq_len, d_out)
        logits = logits_all.mean(dim=1)      # (N, d_out)

        # Add noise to logits (on the same device)
        if noise > 0.0:
            logits = logits + noise * torch.randn_like(logits)

        labels = torch.argmax(logits, dim=-1)

    return x.cpu(), labels.cpu()


# --------------------------------------------------
# Training loop (backprop on student)
# --------------------------------------------------
def bptrain(
    teacher_mha,
    n_train=10000,
    n_test=10000,
    d_model_student=None,
    d_heads_student=None,
    d_in=2,
    d_out=2,
    lr=1e-3,
    batch_size=128,
    n_epochs=50,
    device="cpu",
    seq_len=10,
):
    """
    Train a student MultiHeadAttention to mimic teacher's classification.

    By default, student has same size as teacher, but you can override
    d_model_student and d_heads_student.
    """
    teacher_mha.eval()
    d_model_teacher = teacher_mha.d_model
    d_heads_teacher = teacher_mha.d_heads

    if d_model_student is None:
        d_model_student = d_model_teacher
    if d_heads_student is None:
        d_heads_student = d_heads_teacher

    # 1. Generate datasets
    x_train, y_train = generate_data(teacher_mha, n_train, seq_len, noise=0.0, device=device)
    x_test,  y_test  = generate_data(teacher_mha, n_test,  seq_len, noise=0.0, device=device)

    train_ds = TensorDataset(x_train, y_train)
    test_ds  = TensorDataset(x_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    # 2. Build student MHA
    student = MultiHeadAttention(
        d_model=d_model_student,
        d_heads=d_heads_student,
        d_in=d_in,
        d_out=d_out,
    ).to(device)

    optimizer = optim.Adam(student.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    test_losses = []

    # 3. Training loop
    for epoch in range(n_epochs):
        student.train()
        running_train_loss = 0.0
        n_train_batches = 0

        for xb, yb in train_loader:
            xb = xb.to(device)              # (B, 2)
            yb = yb.to(device)              # (B,)

            seq = x_to_seq(xb, seq_len)          # (B, seq_len, d_in)
            logits_all = student(seq)             # (B, seq_len, d_out)
            logits = logits_all.mean(dim=1)       # (B, d_out)

            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            n_train_batches += 1

        avg_train_loss = running_train_loss / max(1, n_train_batches)
        train_losses.append(avg_train_loss)

        # Evaluate on test set
        student.eval()
        running_test_loss = 0.0
        n_test_batches = 0

        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                seq = x_to_seq(xb, seq_len)      # (B, seq_len, d_in)
                logits_all = student(seq)              # (B, seq_len, d_out)
                logits = logits_all.mean(dim=1)        # (B, d_out)

                loss = criterion(logits, yb)
                running_test_loss += loss.item()
                n_test_batches += 1

        avg_test_loss = running_test_loss / max(1, n_test_batches)
        test_losses.append(avg_test_loss)

        print(
            f"Epoch {epoch+1:03d} / {n_epochs:03d}  "
            f"train_loss = {avg_train_loss:.4f}  test_loss = {avg_test_loss:.4f}"
        )

    return student, train_losses, test_losses





# --------------------------------------------------
# Main
# --------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Teacher maps a length-T_SEQ sinusoidal sequence (parameterized by x in R^2)
    # to a class label; student learns from the same sequences in a teacher-student setup.

    # Teacher dimensions
    d_in = 2
    d_out = 2
    d_heads = 200
    d_model = 20 * d_heads
    T_SEQ = 10


    # Hyperparameters
    n_train = 10000
    n_test = 10000
    lr = 1e-3
    batch_size = 128
    n_epochs = 100

    # 1. Initialize teacher MHA (truth_mha)
    truth_mha = MultiHeadAttention(
        d_model=d_model,
        d_heads=d_heads,
        d_in=d_in,
        d_out=d_out,
    ).to(device)

    # # 2. Optionally visualize teacher decision boundary
    # print("Plotting teacher classification regions...")
    # plot_classification(truth_mha, device, title="Teacher MHA", seq_len=T_SEQ)

    # 3. Train student in teacher–student setup
    student_mha, train_losses, test_losses = bptrain(
        teacher_mha=truth_mha,
        n_train=n_train,
        n_test=n_test,
        d_model_student=None,   # same as teacher by default
        d_heads_student=None,   # same as teacher by default
        d_in=d_in,
        d_out=d_out,
        lr=lr,
        batch_size=batch_size,
        n_epochs=n_epochs,
        device=device,
        seq_len=T_SEQ,
    )

    # 4. Plot train & test loss vs epoch
    epochs = range(1, n_epochs + 1)
    plt.figure(figsize=(5, 4))
    plt.plot(epochs, train_losses, label="train loss")
    plt.plot(epochs, test_losses, label="test loss")
    plt.xlabel("epoch")
    plt.ylabel("cross-entropy loss")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 5. Optionally visualize student decision boundary
    print("Plotting student classification regions...")
    plot_classification(student_mha, device, title="Student MHA", seq_len=T_SEQ)
