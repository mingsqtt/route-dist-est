import numpy as np
import pandas as pd
import dijkstra
import math
import pickle


class DijkstraEstimator:
    def __init__(self, node_file_path, link_file_path, dist_col="dist_corr"):
        self.graph3 = dijkstra.Graph()
        self.graph1 = dijkstra.Graph()
        links_df = pd.read_csv(link_file_path)
        for r, row in links_df.iterrows():
            self.graph1.add_edge(row["frm_node"], row["to_node"], row[dist_col])
            if row["n_link_sample"] >= 3:
                self.graph3.add_edge(row["frm_node"], row["to_node"], row[dist_col])
        nodes_df = pd.read_csv(node_file_path)
        self.node_lat_arr = nodes_df["lat"].values
        self.node_lng_arr = nodes_df["lng"].values
        self.node_names = nodes_df["node"].values
        self.node_col = np.argwhere(nodes_df.columns.values == "node")[0][0]

    def estimate_by_node(self, ori_node, dest_node):
        dijkstra3 = dijkstra.DijkstraSPF(self.graph3, ori_node)
        dist = dijkstra3.get_distance(dest_node)
        if dist < math.inf:
            path = dijkstra3.get_path(dest_node)
            print("\n".join(path))
            print("dist: {}".format(round(dist, 2)))
            return dist, path
        else:
            dijkstra1 = dijkstra.DijkstraSPF(self.graph1, ori_node)
            dist = dijkstra1.get_distance(dest_node)
            if dist < math.inf:
                path = dijkstra1.get_path(dest_node)
                print("\n".join(path))
                print("dist (min1): {}".format(round(dist, 2)))
                return dist, path
            else:
                print("fail")
                return 0.0, None

    def estimate_by_loc(self, ori_lat, ori_lng, dest_lat, dest_lng, alpha=1.1):
        ori_y = self.node_lat_arr - ori_lat
        ori_x = self.node_lng_arr - ori_lng
        ori_nearest_dists = np.round((ori_y ** 2 + ori_x ** 2) ** 0.5 * np.pi * 6371 / 2 / 90, 3)
        ind = np.argmin(ori_nearest_dists)
        ori2node_dist = ori_nearest_dists[ind]
        ori_node = self.node_names[ind]
        dest_y = self.node_lat_arr - dest_lat
        dest_x = self.node_lng_arr - dest_lng
        dest_nearest_dists = np.round((dest_y ** 2 + dest_x ** 2) ** 0.5 * np.pi * 6371 / 2 / 90, 3)
        ind = np.argmin(dest_nearest_dists)
        dest2node_dist = dest_nearest_dists[ind]
        dest_node = self.node_names[ind]
        node2node_dist, path = self.estimate_by_node(ori_node, dest_node)
        if path is not None:
            total_dist = round(ori2node_dist * alpha + node2node_dist + dest2node_dist * alpha, 3)
            center_lat, center_lng = np.mean([ori_lat, dest_lat]), np.mean([ori_lng, dest_lng])
            between_markers = "".join(["|" + node[node.find("@") + 1:] for node in path])
            if (np.abs(ori_lng - dest_lng) > 0.18) or (np.abs(ori_lat - dest_lat) > 0.18):
                url = "https://maps.googleapis.com/maps/api/staticmap?center=Singapore&zoom=11&size=1024x768&maptype=roadmap%20&style=feature:administrative|visibility:off&style=feature:poi|visibility:off&style=feature:landscape|visibility:off&style=feature:road.arterial|visibility:off&key=AIzaSyAq8wmpB8Zdzz__q-j1itlmkJn7IPPloGw&markers=color:green|size:tiny|" + str(
                    ori_lat) + "," + str(ori_lng) + "&markers=color:black|size:tiny" + between_markers + "&markers=color:red|size:tiny|" + str(dest_lat) + "," + str(dest_lng)
            elif (np.abs(ori_lng - dest_lng) > 0.08) or (np.abs(ori_lat - dest_lat) > 0.08):
                url = "https://maps.googleapis.com/maps/api/staticmap?center=" + str(center_lat) + "," + str(
                    center_lng) + "&zoom=12&size=1024x768&maptype=roadmap%20&style=feature:administrative|visibility:off&style=feature:poi|visibility:off&style=feature:landscape|visibility:off&style=feature:road.arterial|visibility:off&key=AIzaSyAq8wmpB8Zdzz__q-j1itlmkJn7IPPloGw&markers=color:green|size:tiny|" + str(
                    ori_lat) + "," + str(ori_lng) + "&markers=color:black|size:tiny" + between_markers + "&markers=color:red|size:tiny|" + str(dest_lat) + "," + str(dest_lng)
            else:
                url = "https://maps.googleapis.com/maps/api/staticmap?center=" + str(center_lat) + "," + str(
                    center_lng) + "&zoom=13&size=1024x768&maptype=roadmap%20&style=feature:administrative|visibility:off&style=feature:poi|visibility:off&style=feature:landscape|visibility:off&style=feature:road.arterial|visibility:off&key=AIzaSyAq8wmpB8Zdzz__q-j1itlmkJn7IPPloGw&markers=color:green|size:small|" + str(
                    ori_lat) + "," + str(ori_lng) + "&markers=color:black|size:small" + between_markers + "&markers=color:red|size:small|" + str(dest_lat) + "," + str(dest_lng)
            return total_dist, node2node_dist, path, url
        else:
            y = ori_lat - dest_lat
            x = ori_lng - dest_lng
            total_dist = np.round((y ** 2 + x ** 2) ** 0.5 * np.pi * 6371 / 2 / 90 * 1.4, 3)
            return total_dist, node2node_dist, path, ""

    def estimate_by_nodes(self, ori_nodes, dest_nodes):
        assert len(ori_nodes) == len(dest_nodes), "ori_nodes, dest_nodes must have same length"
        dist_arr = np.zeros(len(ori_nodes))
        for i in range(len(ori_nodes)):
            ori_node = ori_nodes[i]
            dest_node = dest_nodes[i]
            dijkstra3 = dijkstra.DijkstraSPF(self.graph3, ori_node)
            dist = dijkstra3.get_distance(dest_node)
            if dist < math.inf:
                dist_arr[i] = dist
            else:
                dijkstra1 = dijkstra.DijkstraSPF(self.graph1, ori_node)
                dist = dijkstra1.get_distance(dest_node)
                if dist < math.inf:
                    dist_arr[i] = dist
        return dist_arr

estimator = DijkstraEstimator("data/gps_data_2020-09-01_2020-12-31_valid_trip_data_nodes.csv", "data/gps_data_2020-09-01_2020-12-31_valid_trip_data_links.csv")

estimator.estimate_by_loc(1.318837,103.872236, 1.317065,103.893789)