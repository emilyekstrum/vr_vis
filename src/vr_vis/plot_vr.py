from __future__ import annotations
import os
import pickle as pkl
from typing import Dict, Tuple, List, Union, Any, Optional

import numpy as np
import matplotlib.pyplot as plt
from ripser import ripser


def increase_ball_epsilon(embedding, diameter):
    """ Create Euclidean balls with specified diameter and find overlapping connections.

    Args:
        embedding : (N, d) array
        diameter : float ball diameter

    Returns:
        dict with keys:
            balls, connections, epsilon, diameter, num_connections, h0, h1, h2"""
    
    epsilon = diameter / 2.0

    # Balls for each point
    balls = {
        i: {"center": point, "radius": epsilon, "center_idx": i}
        for i, point in enumerate(embedding)
    }

    # Homology group count using Ripser 
    dgms = ripser(embedding, maxdim=2, thresh=epsilon)["dgms"]
    h0, h1, h2 = len(dgms[0]), len(dgms[1]), len(dgms[2])

    # Form connections when balls overlap
    connections: List[Tuple[int, int]] = []
    n_points = len(embedding)
    for i in range(n_points):
        for j in range(i + 1, n_points):
            if np.linalg.norm(embedding[i] - embedding[j]) <= diameter:
                connections.append((i, j))

    return {
        "balls": balls,
        "connections": connections,
        "epsilon": epsilon,
        "diameter": diameter,
        "num_connections": len(connections),
        "h0": h0,
        "h1": h1,
        "h2": h2,
    }


def plot_filtration_progression(embedding, diameter_range, n_steps = 5):
    """ Plot Vietoris–Rips filtration progression with increasing diameter of Euclidean balls."""
    
    min_d, max_d = diameter_range
    diameters = np.linspace(min_d, max_d, n_steps)

    # uses 3D plot
    if embedding.shape[1] == 2:
        embedding_3d = np.column_stack([embedding, np.zeros(len(embedding))])
        print("Note: Converting 2D embedding to 3D for visualization (z=0)")
    elif embedding.shape[1] >= 3:
        embedding_3d = embedding[:, :3]
        if embedding.shape[1] > 3:
            print(f"Note: Using first 3 dimensions from {embedding.shape[1]}D embedding")
    else:
        raise ValueError("Embedding must have at least 2 dimensions")

    # Two-row grid for plot output
    rows = 2
    cols = int(np.ceil(n_steps / rows))

    fig = plt.figure(figsize=(5 * cols, 12))

    for i, diameter in enumerate(diameters):
        result = increase_ball_epsilon(embedding_3d, diameter)
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")

        centers = np.array([ball["center"] for ball in result["balls"].values()])
        ax.scatter(
            centers[:, 0],
            centers[:, 1],
            centers[:, 2],
            c="red",
            s=5,
            alpha=0.7,
            label="Points",
            edgecolors="red",
            linewidth=1,
        )

        # Plot spheres
        for ball in result["balls"].values():
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            x = ball["center"][0] + ball["radius"] * np.outer(np.cos(u), np.sin(v))
            y = ball["center"][1] + ball["radius"] * np.outer(np.sin(u), np.cos(v))
            z = ball["center"][2] + ball["radius"] * np.outer(np.ones(u.size), np.sin(v))
            ax.plot_surface(
                x, y, z, color="lightblue", alpha=0.02, linewidth=0.3, edgecolor="blue", zorder=1
            )

        # Connections
        if result["connections"]:
            first = True
            for j, k in result["connections"]:
                p1 = result["balls"][j]["center"]
                p2 = result["balls"][k]["center"]
                ax.plot(
                    [p1[0], p2[0]],
                    [p1[1], p2[1]],
                    [p1[2], p2[2]],
                    alpha=0.7,
                    linewidth=1,
                    c="black",
                    zorder=2,
                    label="Connections" if first else "",
                )
                first = False

        ax.set_title(
            f"Diameter: {diameter:.3f}\nH0:{result['h0']}  H1:{result['h1']}  H2:{result['h2']}",
            fontsize=12,
            pad=6,
        )
        ax.set_xlabel("Dimension 1", fontsize=9)
        ax.set_ylabel("Dimension 2", fontsize=9)
        ax.set_zlabel("Dimension 3", fontsize=9)
        if i == 0:
            ax.legend(loc="upper left", fontsize=9)
        ax.set_box_aspect([1, 1, 1])
        ax.grid(True, alpha=0.3)
        ax.view_init(elev=20, azim=45)

    plt.suptitle("Vietoris–Rips Filtration Progression", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.show()

    # Summary stats
    print("Filtration Progression Summary (3D)")
    for i, diameter in enumerate(diameters, start=1):
        result = increase_ball_epsilon(embedding_3d, diameter)
        n = len(embedding_3d)
        max_possible = n * (n - 1) // 2
        density = result["num_connections"] / max_possible if max_possible else 0.0
        sample = result["connections"][:3] if result["connections"] else []
        print(f"Step {i}: d={diameter:.3f}")
        print(f"  → {result['num_connections']} connections")
        print(f"  → Connection density: {density:.4f}")
        print(f"  → Points: {n}")
        if sample:
            print(f"  → Example connections: {sample}")
        print()


def downsample_embedding(embedding, n_samples, sampling_method):
    """ Downsample an embedding to reduce the number of points for faster and clearer visualization.
    
    Args:
        embedding: (N, d) array
        n_samples: target number of points in downsampled embedding
        sampling_method: random, uniform, or first"""
    
    n_points = len(embedding)

    if n_samples is None:
        n_samples = min(500, max(50, int(0.1 * n_points)))

    if n_samples >= n_points:
        return embedding, list(range(n_points))

    if sampling_method == "random":
        indices = np.random.choice(n_points, size=n_samples, replace=False)
    elif sampling_method == "uniform":
        indices = np.linspace(0, n_points - 1, n_samples, dtype=int)
    elif sampling_method == "first":
        indices = np.arange(n_samples)
    else:
        raise ValueError("sampling_method must be 'random', 'uniform', or 'first'")

    indices = np.sort(indices).tolist()
    return embedding[indices], indices


def _validate_and_extract_embedding(embedding_source, mouse_name):
    """ Accepts either a dict or a pickle path that contains
    { <mouse_name>: { 'embedding': np.ndarray, ... }, ... }.
    
    Returns the embedding array for the given mouse_name. """
    
    if isinstance(embedding_source, str):
        if not os.path.exists(embedding_source):
            raise FileNotFoundError(f"Embeddings file not found: {embedding_source}")
        with open(embedding_source, "rb") as f:
            embedding_dict = pkl.load(f)
    elif isinstance(embedding_source, dict):
        embedding_dict = embedding_source
    else:
        raise TypeError("embedding_source must be a dict or a filepath string")

    if mouse_name not in embedding_dict:
        raise ValueError(
            f"Mouse '{mouse_name}' not found. Available: {list(embedding_dict.keys())}"
        )

    emb = embedding_dict[mouse_name].get("embedding")
    if emb is None:
        raise KeyError(f"Entry for '{mouse_name}' lacks key 'embedding'")
    emb = np.asarray(emb)
    if emb.ndim != 2:
        raise ValueError("Embedding must be a 2D array of shape (N, d)")
    return emb


def test_embedding_filtration_workflow(embedding_dict, mouse_name, n_samples: int = 200, diameter_range,
    n_filtration_steps, sampling_method):
    """Tests workflow for embedding visualization and VR filtration."""
        
    original_embedding = _validate_and_extract_embedding(embedding_dict, mouse_name)
    print(f"Original embedding shape: {original_embedding.shape}")

    downsampled, idx = downsample_embedding(
        original_embedding, n_samples=n_samples, sampling_method=sampling_method
    )
    print(f"Downsampled embedding shape: {downsampled.shape}")
    print(f"Selected {len(idx)} / {len(original_embedding)} points")

    print("\nPlot VR Filtration")
    plot_filtration_progression(downsampled, diameter_range, n_filtration_steps)


def run(embedding_source, *, mouse_name = "C155", n_samples = 1000, diameter_range = (0.05, 10.0),
    n_filtration_steps = 10,sampling_method = "uniform"):
    """ Single-call entry point for CLI use.

    Parameters:
        embedding_source : str | dict
            Either a path to a pickle file containing the embeddings dict, or the dict itself.
            Dict should have: { mouse_name: {"embedding": np.ndarray, ...}, ... }
        mouse_name : str
        n_samples : int
        diameter_range : (min, max)
        n_filtration_steps : int
        sampling_method : {'random','uniform','first'}"""
        
    if isinstance(embedding_source, str):
        with open(embedding_source, "rb") as f:
            embedding_dict = pkl.load(f)
    else:
        embedding_dict = embedding_source

    test_embedding_filtration_workflow(
        embedding_dict=embedding_dict,
        mouse_name=mouse_name,
        n_samples=n_samples,
        diameter_range=diameter_range,
        n_filtration_steps=n_filtration_steps,
        sampling_method=sampling_method,
    )
