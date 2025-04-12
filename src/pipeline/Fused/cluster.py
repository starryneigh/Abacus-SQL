import os
import json
import heapq
import torch
import pickle
import argparse
import numpy as np

from typing import List
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon
from transformers import AutoTokenizer, AutoModel, set_seed
from scipy.spatial import ConvexHull, Voronoi


os.environ["TOKENIZERS_PARALLELISM"] = "true"


def encode_sentences_with_transformers(sentences: List[str], model_name: str) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    encoded_input = tokenizer(sentences, padding=True,
                              truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embeddings = model_output.last_hidden_state.mean(dim=1)
    return sentence_embeddings.numpy()


def reduce_examples(examples: List[str], encoder: str) -> List[str]:
    def reduce_points_indices(points: np.ndarray, number: int) -> List[int]:
        """
        Optimized version to return the indices of the reduced points.
        """
        if len(points) <= number:
            return list(range(len(points)))

        # Compute initial full distance matrix
        distances = cdist(points, points)
        np.fill_diagonal(distances, np.inf)

        # Create a min-heap to store distances and point indices
        heap = []
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                heapq.heappush(heap, (distances[i, j], i, j))

        # Set to keep track of removed points
        removed_points = set()

        while len(points) - len(removed_points) > number:
            # Pop the smallest distance pair
            _, point_a, point_b = heapq.heappop(heap)

            # Skip if one of the points was already removed
            if point_a in removed_points or point_b in removed_points:
                continue

            # Determine which point to remove
            min_dist_to_others_a = np.min([distances[point_a, i] for i in range(
                len(points)) if i != point_a and i not in removed_points])
            min_dist_to_others_b = np.min([distances[point_b, i] for i in range(
                len(points)) if i != point_b and i not in removed_points])

            if min_dist_to_others_a < min_dist_to_others_b:
                removed_points.add(point_a)
            else:
                removed_points.add(point_b)

        # Generate list of indices for points that were not removed
        retained_indices = [i for i in range(
            len(points)) if i not in removed_points]
        return retained_indices

    sentence_embeddings = encode_sentences_with_transformers(
        examples, encoder)
    return reduce_points_indices(sentence_embeddings, len(sentence_embeddings[0]))


def optimal_cluster_sentences(sentences: List[str], encoder: str, model_save_path: str = None, max_clusters: int = 8) -> List[List[int]]:
    sentence_embeddings = encode_sentences_with_transformers(
        sentences, encoder)
    best_num_clusters = max_clusters  # TODO
    kmeans = KMeans(n_clusters=best_num_clusters,
                    random_state=0, n_init='auto')
    kmeans.fit(sentence_embeddings)
    if model_save_path:
        pickle.dump(kmeans, open(model_save_path, "wb"))

    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    sentences_with_info = [(sentence, label, np.linalg.norm(embedding - centers[label]), idx)
                           for idx, (sentence, label, embedding) in enumerate(zip(sentences, labels, sentence_embeddings))]
    sentences_with_info.sort(key=lambda x: (x[1], x[2]))
    clustered_sentence_idx = [[] for _ in range(best_num_clusters)]
    for sentence, label, _, idx in sentences_with_info:
        clustered_sentence_idx[label].append(idx)
    return clustered_sentence_idx


def openness_measurement(points: np.ndarray, epsilon: float = 1e-2, measurement: str = "voronoi", pca: PCA = None, hull: ConvexHull = None) -> float:
    def voronoi_measurement(points: np.ndarray, hull: ConvexHull) -> float:
        """
        Compute the maximum distance from each point in the given set to the furthest point in its Voronoi region within the convex hull.

        Parameters:
        points (np.ndarray): A 2D array where each row represents a point (x, y).
        hull (ConvexHull): A ConvexHull object representing the convex hull.

        Returns:
        float: The maximum distance from any point to the furthest point in its Voronoi region within the convex hull.
        """
        if not hull:
            hull = ConvexHull(points)
        # Generate the Voronoi diagram
        vor = Voronoi(points)

        # Create a polygon for the convex hull
        hull_polygon = Polygon(hull.points[hull.vertices])

        max_distance = 0

        for region_index, point_index in enumerate(vor.point_region):
            vertices = vor.regions[point_index]
            if -1 in vertices:
                # Skip regions with a vertex at infinity
                continue

            region_polygon = Polygon([vor.vertices[i] for i in vertices])
            if not hull_polygon.intersects(region_polygon):
                # Skip regions that do not intersect the hull
                continue

            intersection = hull_polygon.intersection(region_polygon)

            if intersection.is_empty:
                continue

            if isinstance(intersection, (Polygon,)):
                points_in_region = np.array(intersection.exterior.coords)
            else:
                points_in_region = np.vstack(
                    [np.array(geom.exterior.coords) for geom in intersection.geoms])

            generator_point = points[region_index]

            for p in points_in_region:
                distance = np.linalg.norm(p - generator_point)
                if distance > max_distance:
                    max_distance = distance

        return max_distance

    if not pca:
        pca = PCA(n_components=2)
        pca = pca.fit(points)
    points = pca.transform(points)

    if measurement == "voronoi":
        return voronoi_measurement(points, hull)
    else:
        raise ValueError(f"Invalid measurement type: {measurement}")


def pro_cluster(data_file, dump_file, encoder_name_or_path, model_dump_file, cluster_number, previous_data_file=None, random_seed=42):
    set_seed(random_seed)

    print("loading data")
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if previous_data_file:
        data += json.load(open(previous_data_file, 'r', encoding='utf-8'))

    # print("reducing examples")
    # data_questions = [d['question'] for d in data]
    # data_index = reduce_examples(data_questions, args.encoder_name_or_path)
    # data = [data[i] for i in data_index]

    print("clustering examples")
    data_questions = [d['question'] for d in data]
    cluster_idx = optimal_cluster_sentences(
        data_questions, encoder_name_or_path, model_dump_file, cluster_number)
    cluster_data = [[data[i] for i in l] for l in cluster_idx]

    print("dumping data")
    with open(dump_file, 'w', encoding='utf-8') as f:
        json.dump(cluster_data, f, ensure_ascii=False, indent=4)


# if __name__ == '__main__':
#     print('start')
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--data_file", type=str, help="data path")
#     parser.add_argument("--dump_file", type=str, help="dump path")
#     parser.add_argument("--previous_data_file", type=str,
#                         help="previous data path")
#     parser.add_argument("--encoder_name_or_path",
#                         type=str, help="encoder path")
#     parser.add_argument("--model_dump_file", type=str, help="model dump path")
#     parser.add_argument("--cluster_number", type=int)
#     parser.add_argument("--random_seed", type=int,
#                         default=42, help="random seed")
#     args = parser.parse_args()
#     set_seed(args.random_seed)

#     print("loading data")
#     with open(args.data_file, 'r', encoding='utf-8') as f:
#         data = json.load(f)
#     if args.previous_data_file:
#         data += json.load(open(args.previous_data_file, 'r', encoding='utf-8'))

#     # print("reducing examples")
#     # data_questions = [d['question'] for d in data]
#     # data_index = reduce_examples(data_questions, args.encoder_name_or_path)
#     # data = [data[i] for i in data_index]

#     print("clustering examples")
#     data_questions = [d['question'] for d in data]
#     cluster_idx = optimal_cluster_sentences(
#         data_questions, args.encoder_name_or_path, args.model_dump_file, args.cluster_number)
#     cluster_data = [[data[i] for i in l] for l in cluster_idx]

#     print("dumping data")
#     with open(args.dump_file, 'w', encoding='utf-8') as f:
#         json.dump(cluster_data, f, ensure_ascii=False, indent=4)
#     # print(
#     #     f"OM : {openness_measurement(encode_sentences_with_transformers(data_questions, args.encoder_name_or_path))}")
