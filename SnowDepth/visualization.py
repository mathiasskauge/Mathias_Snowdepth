import matplotlib.pyplot as plt
import rasterio
import numpy as np

def plot_scatter(y_true, y_pred):
    plt.figure()
    plt.scatter(y_true, y_pred, s=5, alpha=0.3)
    plt.plot([y_true.min(), y_true.max()],
             [y_true.min(), y_true.max()], 'k--')
    plt.xlabel("Observed SD (m)")
    plt.ylabel("Predicted SD (m)")
    plt.title("RF: Observed vs Predicted")
    plt.show()

def plot_map(raster_path, title=""):
    with rasterio.open(raster_path) as src:
        arr = src.read(1)
        plt.figure()
        plt.imshow(arr, cmap='viridis')
        plt.colorbar(label="Snow Depth (m)")
        plt.title(title)
        plt.axis('off')
        plt.show()

def plot_difference(ref_path, pred_path):
    with rasterio.open(ref_path) as src:
        ref = src.read(1)
    with rasterio.open(pred_path) as src:
        pred = src.read(1)
    diff = pred - ref
    plt.figure()
    plt.imshow(diff, cmap='RdBu', vmin=-np.nanmax(np.abs(diff)), vmax=np.nanmax(np.abs(diff)))
    plt.colorbar(label="Predicted − Observed (m)")
    plt.title("Difference Map")
    plt.axis('off')
    plt.show() 