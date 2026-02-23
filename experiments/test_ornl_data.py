import numpy as np

sample_nr = np.load("../datasets/fit_REF_L_194438.npy")

# Check the full R column
q_vals = sample_nr[:, 0]  # Column 0 = Q
r_vals = sample_nr[:, 1]  # Column 1 = R

print(f"Q values:")
print(f"  Min: {q_vals.min():.6f}")
print(f"  Max: {q_vals.max():.6f}")
print(f"  Unique values: {len(np.unique(q_vals))}/{len(q_vals)}")

print(f"\nR values:")
print(f"  Min: {r_vals.min():.6e}")
print(f"  Max: {r_vals.max():.6e}")
print(f"  Unique values: {len(np.unique(r_vals))}/{len(r_vals)}")
print(f"  All same? {np.all(r_vals == r_vals[0])}")

print(f"\nLast 5 rows:")
print(sample_nr[-5:])