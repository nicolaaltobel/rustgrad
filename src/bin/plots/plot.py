import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

base_dir = Path(__file__).resolve().parent
csv_path = base_dir / "moons.csv"
png_path = base_dir / "moons.png"

df = pd.read_csv(csv_path)

grid = df[df["split"] == "grid"].copy()
train = df[df["split"] == "train"]
test = df[df["split"] == "test"]

# decision boundary dalla griglia
grid = grid.sort_values(["x2", "x1"]).reset_index(drop=True)
nx = grid["x1"].nunique()
ny = grid["x2"].nunique()
expected = nx * ny

if len(grid) != expected:
    raise ValueError(f"grid size mismatch: got {len(grid)}, expected {expected}")

xx = grid.x1.values.reshape(ny, nx)
yy = grid.x2.values.reshape(ny, nx)
zz = grid.pred.values.reshape(ny, nx)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, zz, levels=[-999, 0, 999], colors=["#ffaaaa", "#aaaaff"], alpha=0.4)
plt.contour(xx, yy, zz, levels=[0], colors="black", linewidths=1)

# punti train e test
plt.scatter(train.x1, train.x2,
            c=train.label.astype(float),
            cmap="RdBu", vmin=-1, vmax=1, s=10, label="train")
plt.scatter(test.x1, test.x2,
            c=test.label.astype(float),
            cmap="RdBu", vmin=-1, vmax=1, s=40, marker="^", label="test")

plt.legend()
plt.title("Moon dataset — rustgrad")
plt.tight_layout()
plt.savefig(png_path, dpi=150)
plt.show()
print(f"pred range: {grid.pred.min():.4f} to {grid.pred.max():.4f}")
print(f"pred unique: {grid.pred.nunique()}")
print(f"saved plot: {png_path}")