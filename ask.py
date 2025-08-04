import numpy as np
from matplotlib import pyplot as plt

# Set random seed for reproducibility
rng = np.random.default_rng(242)

# Network structure
layers = [3, 8, 2]
layer_x = [0, 1, 2]
neuron_radius = 0.06

# Define base ratios per input for hidden-layer groups,
# group 0–3 in cooler tones, group 5–7 in warmer tones
n_input, n_hidden, n_output = layers
base_inp_hidden0 = rng.uniform(0.0, 0.4, size=n_input)
base_inp_hidden1 = rng.uniform(0.4, 1.0, size=n_input)
# Define base ratios per output for hidden→output groups,
# group 0–3 in cooler tones, group 5–7 in warmer tones
base_hid_out0 = rng.uniform(0.0, 0.4, size=n_output)
base_hid_out1 = rng.uniform(0.4, 1.0, size=n_output)

# Neuron positions (centralize input/output layers)
positions = []
for i, n_neurons in enumerate(layers):
    if n_neurons == 1:
        y = np.array([0.])
    else:
        # Centralize: spread neurons between -a and a, where a = (n_neurons-1)/n_neurons
        a = ((n_neurons - 1) / n_neurons)**2
        y = np.linspace(-0.5 * a, 0.5 * a, n_neurons) * 2
    x = np.full_like(y, layer_x[i], dtype=float)
    positions.append(list(zip(x, y)))

fig, ax = plt.subplots(figsize=(3, 2))
ax.axis('off')
ax.set_aspect('equal')  # Enforce aspect ratio 1

# Draw links with random colors
for l in range(len(layers) - 1):
    for i, (x0, y0) in enumerate(positions[l]):
        for j, (x1, y1) in enumerate(positions[l+1]):
            # Use RdBu colormap to pick a random color for each link
            cmap = plt.get_cmap('RdBu')
            # Group-based color assignments with per-node base + noise
            if l == 0:
                # input → hidden: same base for hidden neurons 0–3 or 5–7 per input i
                if j < 4:
                    base = base_inp_hidden0[i]
                elif j > 4:
                    base = base_inp_hidden1[i]
                else:
                    base = rng.uniform(0, 1)  # neuron 4 random
            else:
                # hidden → output: same base for hidden neurons 0–3 or 5–7 per output j
                if i < 4:
                    base = base_hid_out0[j]
                elif i > 4:
                    base = base_hid_out1[j]
                else:
                    base = rng.uniform(0, 1)  # neuron 4 random
            ratio = np.clip(base + rng.normal(0, 0.05), 0, 1)
            color = cmap(ratio)
            ax.plot([x0, x1], [y0, y1], color=color, linewidth=2, alpha=0.9)

# Draw neurons
for l, layer in enumerate(positions):
    for (x, y) in layer:
        circle = plt.Circle((x, y), neuron_radius, color='w', ec='k', linewidth=1.5, zorder=3)
        ax.add_patch(circle)

plt.tight_layout()
plt.savefig('NN_trained.pdf', format='pdf', bbox_inches='tight', pad_inches=0)
plt.show()