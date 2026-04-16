import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm


AGENT_TYPES = [
    "vehicle",
    "pedestrian",
    "motorcyclist",
    "cyclist",
    "bus",
    "static",
    "background",
    "construction",
    "riderless_bicycle",
    "unknown",
]


def to_numpy(array):
    if isinstance(array, np.ndarray):
        return array
    return array.cpu().numpy()


def local_future_trajs(position_xy, heading, num_historical_steps, num_future_steps):
    origin = position_xy[:, num_historical_steps - 1:num_historical_steps, :]
    future_xy = position_xy[:, num_historical_steps:num_historical_steps + num_future_steps, :] - origin
    theta = heading[:, num_historical_steps - 1]

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rot_mat = np.zeros((position_xy.shape[0], 2, 2), dtype=np.float32)
    rot_mat[:, 0, 0] = cos_theta
    rot_mat[:, 0, 1] = -sin_theta
    rot_mat[:, 1, 0] = sin_theta
    rot_mat[:, 1, 1] = cos_theta

    return np.einsum("ntc,ncd->ntd", future_xy, rot_mat)


def parse_args():
    parser = argparse.ArgumentParser(description="Cluster AV2 val trajectories per agent type in local frame")
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--val_processed_dir", type=str, default=None)
    parser.add_argument("--num_historical_steps", type=int, default=50)
    parser.add_argument("--num_future_steps", type=int, default=60)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--save_dir", type=str, default="data/kmeans")
    parser.add_argument("--vis_dir", type=str, default="vis/kmeans")
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    processed_dir = Path(args.val_processed_dir) if args.val_processed_dir else Path(args.root) / "val" / "processed"
    pkl_files = sorted(list(processed_dir.glob("*.pkl")) + list(processed_dir.glob("*.pickle")))
    if len(pkl_files) == 0:
        raise ValueError(f"No pkl files found in {processed_dir}")

    intentions = {i: [] for i in range(len(AGENT_TYPES))}
    class_counts = np.zeros(len(AGENT_TYPES), dtype=np.int64)

    for pkl_path in tqdm(pkl_files, desc="Loading val pkl files"):
        with open(pkl_path, "rb") as handle:
            data = pickle.load(handle)

        agent = data["agent"]
        agent_type = to_numpy(agent["type"]).astype(np.int64)
        valid_mask = to_numpy(agent["valid_mask"]).astype(bool)
        position = to_numpy(agent["position"]).astype(np.float32)
        heading = to_numpy(agent["heading"]).astype(np.float32)

        current_valid = valid_mask[:, args.num_historical_steps - 1]
        future_valid = valid_mask[:, args.num_historical_steps:args.num_historical_steps + args.num_future_steps].all(axis=1)
        keep_mask = current_valid & future_valid
        if not keep_mask.any():
            continue

        position = position[keep_mask, :, :2]
        heading = heading[keep_mask]
        agent_type = agent_type[keep_mask]
        trajs_local = local_future_trajs(
            position_xy=position,
            heading=heading,
            num_historical_steps=args.num_historical_steps,
            num_future_steps=args.num_future_steps,
        )

        for class_idx in range(len(AGENT_TYPES)):
            cls_mask = agent_type == class_idx
            if not cls_mask.any():
                continue
            cls_trajs = trajs_local[cls_mask]
            intentions[class_idx].append(cls_trajs)
            class_counts[class_idx] += cls_trajs.shape[0]

    clusters = np.zeros((len(AGENT_TYPES), args.k, args.num_future_steps, 2), dtype=np.float32)
    for class_idx, class_name in enumerate(AGENT_TYPES):
        if class_counts[class_idx] < args.k:
            raise ValueError(
                f"Class {class_name} has {class_counts[class_idx]} samples, but k={args.k}."
            )

        cls_trajs = np.concatenate(intentions[class_idx], axis=0)
        cls_flat = cls_trajs.reshape(cls_trajs.shape[0], -1)
        kmeans = KMeans(n_clusters=args.k, random_state=args.seed, n_init="auto")
        centers = kmeans.fit(cls_flat).cluster_centers_.reshape(args.k, args.num_future_steps, 2)
        clusters[class_idx] = centers

        plt.figure(figsize=(6, 6))
        for center in centers:
            plt.plot(center[:, 0], center[:, 1], marker="o", markersize=2)
        plt.scatter([0.0], [0.0], c="red", s=20)
        plt.axis("equal")
        plt.title(f"motion_intention_{class_name}_k{args.k}")
        plt.xlabel("x_local")
        plt.ylabel("y_local")
        Path(args.vis_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(Path(args.vis_dir) / f"motion_intention_val_{class_name}_{args.k}.png", bbox_inches="tight")
        plt.close()

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    np.save(Path(args.save_dir) / f"kmeans_motion_val_{args.k}.npy", clusters)

    print("Finished clustering on val split.")
    for class_idx, class_name in enumerate(AGENT_TYPES):
        print(f"{class_name}: {class_counts[class_idx]} samples")
    print(f"Saved centers to {Path(args.save_dir) / f'kmeans_motion_val_{args.k}.npy'}")


if __name__ == "__main__":
    main()