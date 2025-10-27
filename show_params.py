"""
Script to display network parameters from the trained policy.
"""
import torch

# Load the saved model parameters
state_dict = torch.load('cartpole_reinforce_policy.pth')

print("=" * 80)
print("TRAINED POLICY NETWORK PARAMETERS")
print("=" * 80)
print()

# Iterate through all parameters in the state dict
for param_name, param_value in state_dict.items():
    print(f"Parameter: {param_name}")
    print(f"Shape: {param_value.shape}")
    print(f"Values:\n{param_value}")
    print("-" * 80)
    print()

# Print summary statistics
print("=" * 80)
print("SUMMARY")
print("=" * 80)
total_params = sum(p.numel() for p in state_dict.values())
print(f"Total number of parameters: {total_params}")
print()

for param_name, param_value in state_dict.items():
    print(f"{param_name}:")
    print(f"  Shape: {param_value.shape}")
    print(f"  Size: {param_value.numel()} parameters")
    print(f"  Min: {param_value.min().item():.6f}")
    print(f"  Max: {param_value.max().item():.6f}")
    print(f"  Mean: {param_value.mean().item():.6f}")
    print(f"  Std: {param_value.std().item():.6f}")
    print()
