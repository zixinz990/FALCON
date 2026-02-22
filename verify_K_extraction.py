"""Verify that K_nn(z) == K @ z + b for random inputs."""

import torch
from humanoid_linear_distill.utils.networks import KNet

# Create a KNet with default dimensions
feature_dim = 1024
action_dim = 29
K_nn = KNet(input_size=feature_dim, output_size=action_dim)
K_nn.eval()

# Extract K and b
K = K_nn.linear1.weight.data.clone()  # (action_dim, feature_dim)
b = K_nn.linear1.bias.data.clone()    # (action_dim,)

# Test with random inputs
torch.manual_seed(42)
for i in range(5):
    z = torch.randn(1, feature_dim)  # single input
    out_nn = K_nn(z)
    out_mat = z @ K.T + b
    diff = (out_nn - out_mat).abs().max().item()
    print(f"Test {i+1} (single):  max abs diff = {diff:.2e}")

# Test with a batch
z_batch = torch.randn(32, feature_dim)
out_nn_batch = K_nn(z_batch)
out_mat_batch = z_batch @ K.T + b
diff_batch = (out_nn_batch - out_mat_batch).abs().max().item()
print(f"Test batch (32):    max abs diff = {diff_batch:.2e}")

print("\nAll diffs should be 0.00e+00 (or < 1e-7 due to float precision).")
