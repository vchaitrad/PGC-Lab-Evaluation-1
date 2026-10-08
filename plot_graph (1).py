import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sizes = ["1M", "10M", "50M"]
serial = [0.588, 10.050, 51.485]
omp4 = [0.659, 7.509, 43.498]
cuda_kernel = [0.051, 0.459, 2.284]
cuda_total = [2.690, 25.964, 131.537]

x = np.arange(len(sizes))
w = 0.2
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - 1.5 * w, serial, w, label="Serial CPU", color="#6B7280")
ax.bar(x - 0.5 * w, omp4, w, label="OpenMP (4 threads)", color="#1F6FEB")
ax.bar(x + 0.5 * w, cuda_kernel, w, label="CUDA kernel only", color="#76B900")
ax.bar(x + 1.5 * w, cuda_total, w, label="CUDA total (with copy)", color="#F59E0B")
ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels(sizes)
ax.set_xlabel("Array size (N)")
ax.set_ylabel("Execution time (ms, log scale)")
ax.set_title("CPU vs GPU Vector Addition")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("graph.png", dpi=150)
print("saved graph.png")
