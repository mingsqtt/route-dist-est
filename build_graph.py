from lmpylib.core import *
from lmpylib.geo import plt_trajectory
import pickle
from sklearn.cluster import AgglomerativeClustering
from numpy.linalg import norm
import dijkstra

with open("data/gps_data_2020-09-01_2020-12-31_valid_mov_session.pickle", "br") as f:
    gps_data = pickle.load(f)

prev_lat = np.array(shift_down(gps_data["lat"]))
prev_lng = np.array(shift_down(gps_data["lng"]))
y1 = gps_data["lat"].values - prev_lat
x1 = gps_data["lng"].values - prev_lng
prev_sess = np.array(shift_down(gps_data["mov_sess"], fill_head_with=None))
diff_sess_filt = gps_data["mov_sess"] != prev_sess
y1[diff_sess_filt] = 0
x1[diff_sess_filt] = 0

next_lat = np.array(shift_up(gps_data["lat"]))
next_lng = np.array(shift_up(gps_data["lng"]))
y2 = next_lat - gps_data["lat"].values
x2 = next_lng - gps_data["lng"].values
next_sess = np.array(shift_up(gps_data["mov_sess"], fill_tail_with=None))
diff_sess_filt = gps_data["mov_sess"] != next_sess
y2[diff_sess_filt] = 0
x2[diff_sess_filt] = 0

next_dist = np.array(shift_up(gps_data["dist"]))
next_dist[diff_sess_filt] = 0
gps_data["next_dist"] = next_dist

next_next_sess = np.array(shift_up(next_sess, fill_tail_with=None))
next_next_lat = np.array(shift_up(next_lat))
next_next_lng = np.array(shift_up(next_lng))
y3 = next_next_lat - gps_data["lat"].values
x3 = next_next_lng - gps_data["lng"].values

sim2 = (y1 * y2 + x1 * x2) / ((y1 ** 2 + x1 ** 2) ** 0.5 * (y2 ** 2 + x2 ** 2) ** 0.5)
sim2[pd.isna(sim2)] = 1

sim3 = (y1 * y3 + x1 * x3) / ((y1 ** 2 + x1 ** 2) ** 0.5 * (y3 ** 2 + x3 ** 2) ** 0.5)
sim3[pd.isna(sim3)] = 1

orient_sim = sim2
same_sess_filt = gps_data["mov_sess"] == next_next_sess
override_filt = same_sess_filt & (next_dist < 0.03)
orient_sim[override_filt] = np.minimum(sim2[override_filt], sim3[override_filt])

gps_data["orient_sim"] = orient_sim

# sample_sess_ids = np.random.choice(gps_data["mov_sess"].unique(), 1)
# sample_gps_data = gps_data.loc[gps_data["mov_sess"].isin(sample_sess_ids), :]
# filt = (np.abs(sample_gps_data["orient_sim"]) < 0.2) & (sample_gps_data["dist"] > 0.1)
# plot(sample_gps_data["lng"].values[filt], sample_gps_data["lat"].values[filt], marker_size=200, style="scatter", color="black")
# plt_trajectory(sample_gps_data["lat"].values, sample_gps_data["lng"].values, sample_gps_data["mov_sess"].values)

gps_data["node"] = None
gps_data["node_lat"] = np.nan
gps_data["node_lng"] = np.nan

global_from_lat, global_from_lng = 1.2, 103.6
global_to_lat, global_to_lng = 1.48, 104.1
# corner_point = (global_from_lat, global_from_lng), (global_to_lat, global_to_lng)
unit_deg_1km = np.round(((1 * 90 * 2 / 6371 / np.pi) ** 2) ** 0.5, 3)
unit_deg_0p2km = np.round(((0.2 * 90 * 2 / 6371 / np.pi) ** 2) ** 0.5, 3)
nrow = int(np.ceil((global_to_lat - global_from_lat) / unit_deg_1km))
ncol = int(np.ceil((global_to_lng - global_from_lng) / unit_deg_1km))
lats = np.round(np.linspace(global_from_lat, global_from_lat + nrow * unit_deg_1km, nrow + 1), 3)
longs = np.round(np.linspace(global_from_lng, global_from_lng + ncol * unit_deg_1km, ncol + 1), 3)
grids_frm = cartesian_array(lats, longs, return_as_dataframe=True).values
grids = np.concatenate([grids_frm, grids_frm + unit_deg_1km], axis=1)


def map_to_processing_region(test_point):
    lat, lng = test_point[0], test_point[1]
    cell_y = int((lat - global_from_lat) / unit_deg_1km)
    cell_x = int((lng - global_from_lng) / unit_deg_1km)
    cell_from_lat, cell_to_lat = np.round(global_from_lat + (cell_y * unit_deg_1km), 3), np.round(global_from_lat + ((cell_y + 1) * unit_deg_1km), 3)
    cell_from_lng, cell_to_lng = np.round(global_from_lng + (cell_x * unit_deg_1km), 3), np.round(global_from_lng + ((cell_x + 1) * unit_deg_1km), 3)
    center_from_lat, center_to_lat = cell_from_lat + unit_deg_0p2km, cell_to_lat - unit_deg_0p2km
    center_from_lng, center_to_lng = cell_from_lng + unit_deg_0p2km, cell_to_lng - unit_deg_0p2km
    print("{}-{}".format(cell_y, cell_x))
    if (center_from_lat <= lat <= center_to_lat) and (center_from_lng <= lng <= center_to_lng):
        print("center")
    elif lng < center_from_lng:
        print("{}-{}".format(cell_y, cell_x - 1))
        if center_from_lat <= lat <= center_to_lat:
            print("left")
        elif lat < center_from_lat:
            print("bottom-left")
            print("{}-{}".format(cell_y - 1, cell_x))
            print("{}-{}".format(cell_y - 1, cell_x - 1))
        else:
            print("top-left")
            print("{}-{}".format(cell_y + 1, cell_x))
            print("{}-{}".format(cell_y + 1, cell_x - 1))
    elif lng > center_to_lng:
        print("{}-{}".format(cell_y, cell_x + 1))
        if center_from_lat <= lat <= center_to_lat:
            print("right")
        elif lat < center_from_lat:
            print("bottom-right")
            print("{}-{}".format(cell_y - 1, cell_x))
            print("{}-{}".format(cell_y - 1, cell_x + 1))
        else:
            print("top-right")
            print("{}-{}".format(cell_y + 1, cell_x))
            print("{}-{}".format(cell_y + 1, cell_x + 1))
    elif lat < center_from_lat:
        print("bottom")
        print("{}-{}".format(cell_y - 1, cell_x))
    else:
        print("top")
        print("{}-{}".format(cell_y + 1, cell_x))
# lat, lng = 1.213, 103.623 # center
# lat, lng = 1.213, 103.619 # left
# lat, lng = 1.213, 103.626 # right
# lat, lng = 1.217, 103.623 # top
# lat, lng = 1.210, 103.623 # bottom
# lat, lng = 1.217, 103.619 # top-left
# lat, lng = 1.210, 103.619 # bottom-left
# lat, lng = 1.217, 103.626 # top-right
# lat, lng = 1.210, 103.626 # bottom-right


def orient_sim_hist(test_point, q=0.1):
    inner_grid_filt = (grids[:, 0] < test_point[0]) & (grids[:, 2] > test_point[0]) & (grids[:, 1] < test_point[1]) & (
                grids[:, 3] > test_point[1])
    inner_grid = grids[np.argwhere(inner_grid_filt)[0, 0]]
    outer_grid = inner_grid.copy()
    outer_grid[:2] = outer_grid[:2] - unit_deg_0p2km
    outer_grid[2:] = outer_grid[2:] + unit_deg_0p2km
    values = gps_data.loc[(gps_data["lat"] > outer_grid[0]) & (gps_data["lat"] < outer_grid[2]) & (gps_data["lng"] > outer_grid[1]) & (
                gps_data["lng"] < outer_grid[3]), "orient_sim"]
    hist(values, bins=20)
    values = np.abs(values[values > -0.5])
    print(np.quantile(values, q))


def analyse_grid_cell(test_point, orient_sim_thresh="auto", dist_thresh=0.15, mk_size=0.01, clust1_dist_thresh=0.001, clust2_dist_thresh=0.001, min_clust_size=5, show_plot=True):
    map_to_processing_region(test_point)
    inner_grid_filt = (grids[:, 0] < test_point[0]) & (grids[:, 2] > test_point[0]) & (grids[:, 1] < test_point[1]) & (grids[:, 3] > test_point[1])
    inner_grid = grids[np.argwhere(inner_grid_filt)[0, 0]]
    outer_grid = inner_grid.copy()
    outer_grid[:2] = outer_grid[:2] - unit_deg_0p2km
    outer_grid[2:] = outer_grid[2:] + unit_deg_0p2km
    outer_gps_filt = (gps_data["lat"] >= outer_grid[0]) & (gps_data["lat"] < outer_grid[2]) & (gps_data["lng"] >= outer_grid[1]) & (gps_data["lng"] < outer_grid[3])
    if np.sum(outer_gps_filt) < min_clust_size:
        return

    outer_gps_data = gps_data.loc[outer_gps_filt, ["lat", "lng", "orient_sim", "dist", "next_dist"]].copy()
    print("{} data points within outer grid".format(len(outer_gps_data)))
    if show_plot:
        plot(outer_gps_data["lng"], outer_gps_data["lat"], marker_size=mk_size, color="silver", style="scatter", x_scale_ticks=[],
             y_scale_ticks=[], show=True)

    orient_vals = outer_gps_data["orient_sim"]
    if orient_sim_thresh == "auto":
        orient_sim_thresh = min(np.quantile(np.abs(orient_vals[orient_vals > -0.5]), 0.1), 0.85)
        print("auto orient_sim_thresh: {}".format(orient_sim_thresh))
    conj_filt = (orient_vals > -0.5) & (np.abs(orient_vals) < orient_sim_thresh) & (outer_gps_data["dist"] > dist_thresh)
    conj_points = np.stack([outer_gps_data["lng"].values[conj_filt], outer_gps_data["lat"].values[conj_filt]], axis=1)
    print("{} conj points for sim matrix".format(len(conj_points)))
    if len(conj_points) < min_clust_size:
        return

    clusters = AgglomerativeClustering(n_clusters=None, affinity="euclidean", distance_threshold=clust1_dist_thresh,
                                       linkage="complete").fit(conj_points)
    clust_lv1 = pd.DataFrame(
        {
            "lng": conj_points[:, 0],
            "lat": conj_points[:, 1],
            "c1": clusters.labels_.astype(str)
        }
    )
    outer_gps_data["c1"] = None
    outer_gps_data.loc[conj_filt, "c1"] = clust_lv1["c1"].values

    clust_size = pd.DataFrame({"c1": clust_lv1["c1"], "c1_size": 1}).groupby("c1", as_index=False).sum()
    valid_clust = clust_size.loc[clust_size["c1_size"] >= max(min(min_clust_size, np.max(clust_size["c1_size"])), 2), "c1"]
    clust_lv1 = clust_lv1.loc[clust_lv1["c1"].isin(valid_clust), :]
    outer_gps_data.loc[outer_gps_data["c1"].isin(valid_clust) == False, "c1"] = None
    n_lv1_clust = len(valid_clust)
    print("{} lv1 clusters".format(n_lv1_clust))

    clust_lv2 = clust_lv1[["c1", "lat", "lng"]].groupby("c1", as_index=False).mean()
    if n_lv1_clust > 1:
        clusters_cons = AgglomerativeClustering(n_clusters=None, affinity="euclidean", distance_threshold=clust2_dist_thresh,
                                                linkage="complete").fit(clust_lv2[["lng", "lat"]].values)
        clust_lv2["c2"] = clusters_cons.labels_.astype(str)
        outer_gps_data = outer_gps_data.merge(clust_lv2[["c1", "c2"]], how="left", on="c1")
        clust_lv2 = clust_lv2.merge(clust_size, how="inner", on="c1")
        clust_lv2 = clust_lv2.merge(clust_lv2[["c2", "c1_size"]].groupby("c2", as_index=False).sum(), how="inner", on="c2")
        clust_lv2["weighted"] = clust_lv2["c1_size_x"] / clust_lv2["c1_size_y"]
        clust_lv2["c_lat"] = clust_lv2["lat"] * clust_lv2["weighted"]
        clust_lv2["c_lng"] = clust_lv2["lng"] * clust_lv2["weighted"]
        clust_lv2 = clust_lv2[["c2", "c1_size_y", "c_lat", "c_lng"]].groupby(["c2", "c1_size_y"], as_index=False).sum()
        clust_lv2.columns = ["c2", "clust_size", "c_lat", "c_lng"]
    else:
        outer_gps_data["c2"] = outer_gps_data["c1"]
        clust_lv2.columns = ["c2", "c_lat", "c_lng"]
        clust_lv2.insert(1, "clust_size", clust_size["c1_size"].head(n_lv1_clust))

    clust_lv2["c_lat"] = np.round(clust_lv2["c_lat"], 6)
    clust_lv2["c_lng"] = np.round(clust_lv2["c_lng"], 6)
    print("{} lv2 clusters".format(len(clust_lv2)))
    clust_lv2["node"] = None
    node_filt = (clust_lv2["c_lat"] >= inner_grid[0]) & (clust_lv2["c_lat"] < inner_grid[2]) & (clust_lv2["c_lng"] >= inner_grid[1]) & (clust_lv2["c_lng"] < inner_grid[3])
    print("{} nodes".format(np.sum(node_filt)))
    clust_lv2.loc[node_filt, "node"] = clust_lv2.loc[node_filt, "c2"] + "@" + clust_lv2.loc[node_filt, "c_lat"].astype(str) + "," + clust_lv2.loc[node_filt, "c_lng"].astype(str)
    if show_plot:
        plot(clust_lv2.loc[node_filt, ["c_lng", "c_lat", "c2"]], marker_size=40, style="scatter", marker="x", group_by="c2", legend_loc=None)
        plot(clust_lv2.loc[node_filt == False, ["c_lng", "c_lat", "c2"]], marker_size=40, style="scatter", marker="o", group_by="c2", legend_loc=None)
    outer_gps_data = outer_gps_data.merge(clust_lv2, how="left", on="c2")

    existing_vals = gps_data.loc[outer_gps_filt, ["node", "node_lat", "node_lng"]].values
    existing_filt = pd.isna(existing_vals[:, 0]) == False
    outer_gps_data.loc[existing_filt, ["node", "c_lat", "c_lng"]] = existing_vals[existing_filt, :]
    gps_data.loc[outer_gps_filt, ["node", "node_lat", "node_lng"]] = outer_gps_data[["node", "c_lat", "c_lng"]].values

    if not show_plot:
        return clust_lv2.loc[node_filt, :]


test_pt = (1.3348805606340475, 103.69643382382048)
# orient_sim_hist(test_pt)
# test_point = test_pt
analyse_grid_cell(test_pt, mk_size=0.1)

test_pt = (1.3349039435624175, 103.70170307419603)
# test_point = test_pt
analyse_grid_cell(test_pt, mk_size=0.1)

test_pt = (1.3378904290603884, 103.70284148968793)
# test_point = test_pt
analyse_grid_cell(test_pt, mk_size=0.1)

test_pt = (1.340765640239507, 103.69678662168072)
# test_point = test_pt

analyse_grid_cell(test_pt, mk_size=0.021)

test_pt = (1.3210113517087954, 103.6492186757177)
# orient_sim_hist(test_pt)
analyse_grid_cell(test_pt)

test_pt = (1.2823219396175427, 103.62274168337126)
analyse_grid_cell(test_pt)

test_pt = (1.3101204581760328, 103.71720540409254)
analyse_grid_cell(test_pt)

test_pt = (1.3110065251626433, 103.69773228791186)
analyse_grid_cell(test_pt)

test_pt = (1.3054235401016214, 103.72281858220143)
analyse_grid_cell(test_pt)

test_pt = (1.2764294602407502, 103.76499457830627)
analyse_grid_cell(test_pt)

test_pt = (1.4156028166358272, 103.80417248054756)
analyse_grid_cell(test_pt, mk_size=0.03)

test_pt = (1.422904084331364, 103.75682614146433)
analyse_grid_cell(test_pt, mk_size=0.03)

test_pt = (1.3634386148574054, 103.70695837658268)
# orient_sim_hist(test_pt)
analyse_grid_cell(test_pt, mk_size=0.03)

test_pt = (1.3310858512447215, 103.8633086867684)
analyse_grid_cell(test_pt, mk_size=0.03)

test_pt = (1.3402331606867754, 103.77859816219107)
# orient_sim_hist(test_pt)
analyse_grid_cell(test_pt)

test_pt = (1.3518595575908867, 103.71185700890666)
# test_point = test_pt
analyse_grid_cell(test_pt)


all_nodes = list()
for c in range(len(grids)):
    test_pt = (np.mean(grids[c, [0, 2]]), np.mean(grids[c, [1, 3]]))
    print("======= {} =======\n{}".format(c, test_pt))
    nodes = analyse_grid_cell(test_pt, show_plot=False)
    all_nodes.append(nodes)
    # analyse_grid_cell(test_pt)
    print()
nodes_df = pd.concat(all_nodes, ignore_index=True)
nodes_df.columns = ["c", "clust_size", "lat", "lng", "node"]
plot(nodes_df[["lng", "lat", "c"]], marker_size=0.5, style="scatter", marker="x", group_by="c", legend_loc=None, x_scale_ticks=[], y_scale_ticks=[])
len(nodes_df)

sess_node_count = gps_data[["mov_sess", "node"]].groupby("mov_sess", as_index=False).count()
sess_one_node = sess_node_count.loc[sess_node_count["node"] == 1, :]
sess_multi_nodes = sess_node_count.loc[sess_node_count["node"] > 1, :]
print("{} out of {} trips with at least 1 node".format(len(sess_one_node) + len(sess_multi_nodes), len(sess_node_count)))
print("{} out of {} trips with >1 node".format(len(sess_multi_nodes), len(sess_one_node) + len(sess_multi_nodes)))

gps_data["inferred_node"] = None
gps_data["inferred_node_lat"] = np.nan
gps_data["inferred_node_lng"] = np.nan
# 6,8,13,21
for i, sess_id in enumerate(sess_node_count["mov_sess"]):
    # sess_id = 98584
    sess_data = gps_data.loc[gps_data["mov_sess"] == sess_id, ["lat", "lng"]]

    mtx_node2pt_lat = np.repeat(nodes_df["lat"].values, len(sess_data)).reshape((-1, len(sess_data)))
    mtx_node2pt_lng = np.repeat(nodes_df["lng"].values, len(sess_data)).reshape((-1, len(sess_data)))
    mtx_pt2node_lat = np.repeat(sess_data["lat"].values.reshape((-1, 1)), len(nodes_df), axis=1).transpose()
    mtx_pt2node_lng = np.repeat(sess_data["lng"].values.reshape((-1, 1)), len(nodes_df), axis=1).transpose()
    y = mtx_node2pt_lat - mtx_pt2node_lat
    x = mtx_node2pt_lng - mtx_pt2node_lng
    mtx_km = np.round((x ** 2 + y ** 2) ** 0.5 * np.pi * 6371 / 2 / 90, 3)
    nearest_node_indices = np.argmin(mtx_km, axis=0)
    shortest_dists = mtx_km[nearest_node_indices, range(len(nearest_node_indices))]
    matched_pt_indices = np.argwhere(shortest_dists < 0.03).flatten()
    if len(matched_pt_indices) > 0:
        gps_data.loc[sess_data.index[matched_pt_indices], ["inferred_node", "inferred_node_lat", "inferred_node_lng"]] = nodes_df.loc[
            nearest_node_indices[matched_pt_indices], ["node", "lat", "lng"]].values

    if i % 100 == 0:
        print("[{}]".format(i))

with open("data/inferred_node_np_30m_infer.pickle", "br") as f:
    k = pickle.load(f)
    gps_data["node"] = k[:, 0]
    gps_data["node_lat"] = k[:, 1]
    gps_data["node_lng"] = k[:, 2]
    gps_data["inferred_node"] = k[:, 3]
    gps_data["inferred_node_lat"] = k[:, 4]
    gps_data["inferred_node_lng"] = k[:, 5]
    gps_data["eff_node"] = k[:, 0]
    gps_data["eff_node_lat"] = k[:, 1]
    gps_data["eff_node_lng"] = k[:, 2]
filt = pd.isna(gps_data["eff_node"])
gps_data.loc[filt, "eff_node"] = gps_data.loc[filt, "inferred_node"].values
gps_data.loc[filt, "eff_node_lat"] = gps_data.loc[filt, "inferred_node_lat"].values
gps_data.loc[filt, "eff_node_lng"] = gps_data.loc[filt, "inferred_node_lng"].values

nodes_df = pd.read_csv("data/nodes_df_30m_infer.csv")


sess_node_count = gps_data[["mov_sess", "eff_node"]].groupby("mov_sess", as_index=False).count()
sess_one_node = sess_node_count.loc[sess_node_count["eff_node"] == 1, :]
sess_multi_nodes = sess_node_count.loc[sess_node_count["eff_node"] > 1, :]
print("{} out of {} trips with at least 1 node".format(len(sess_one_node) + len(sess_multi_nodes), len(sess_node_count)))
print("{} out of {} trips with >1 node".format(len(sess_multi_nodes), len(sess_one_node) + len(sess_multi_nodes)))


gps_data["dist_corr"] = gps_data["dist"] * 1.1
filt = (gps_data["time_del"] > 3*60)
gps_data.loc[filt, "dist_corr"] = gps_data.loc[filt, "dist"] * 1.4
links_frm_pt_lat = list()
links_frm_pt_lng = list()
links_frm_node = list()
links_to_pt_lat = list()
links_to_pt_lng = list()
links_to_node = list()
links_dist = list()
links_dist_corr = list()
links_time = list()
links_sess = list()
summary(gps_data.loc[(pd.isna(gps_data["eff_node"]) == False) & (gps_data["dist"] > 10), "dist"])
for i, sess_id in enumerate(sess_multi_nodes["mov_sess"]):
    # sess_id = 98584
    # sess_id = 13
    sess_data = gps_data.loc[gps_data["mov_sess"] == sess_id, ["dist", "time_del", "dist_corr", "lat", "lng", "eff_node", "eff_node_lat", "eff_node_lng", "node", "inferred_node"]]
    node_indices = np.argwhere(pd.isna(sess_data["eff_node"].values) == False).flatten()
    acc_dist = 0
    for n in range(len(node_indices) - 1):
        frm = node_indices[n]
        to = node_indices[n + 1]
        frm_loc = sess_data.iloc[frm, [3, 4, 5]].values
        to_loc = sess_data.iloc[to, [3, 4, 5]].values
        if frm_loc[2] == to_loc[2]:
            continue
        dist_sum = np.sum(sess_data.iloc[frm + 1:to + 1, 0])
        time_sum = np.sum(sess_data.iloc[frm + 1:to + 1, 1])
        dist_corr_sum = np.sum(sess_data.iloc[frm + 1:to + 1, 2])
        acc_dist += dist_corr_sum
        # print(np.round(frm_loc[:2].astype(float), 7).tolist(), np.round(to_loc[:2].astype(float), 7).tolist(), np.round(dist_sum, 1), np.round(acc_dist, 1), to_loc[2])
        links_frm_pt_lat.append(frm_loc[0])
        links_frm_pt_lng.append(frm_loc[1])
        links_frm_node.append(frm_loc[2])
        links_to_pt_lat.append(to_loc[0])
        links_to_pt_lng.append(to_loc[1])
        links_to_node.append(to_loc[2])
        links_dist.append(dist_sum)
        links_time.append(time_sum)
        links_dist_corr.append(dist_corr_sum)
        links_sess.append(sess_id)
    if i % 100 == 0:
        print("[{}]".format(i))


links_raw = pd.DataFrame({
    "frm_pt_lat": links_frm_pt_lat,
    "frm_pt_lng": links_frm_pt_lng,
    "frm_node": links_frm_node,
    "to_pt_lat": links_to_pt_lat,
    "to_pt_lng": links_to_pt_lng,
    "to_node": links_to_node,
    "dist": links_dist,
    "dist_corr": links_dist_corr,
    "time": links_time,
    "mov_sess": links_sess,
})
links_raw["link"] = links_raw["frm_node"] + "-" + links_raw["to_node"]
filt = links_raw["frm_node"] == links_raw["to_node"]
links_raw_same = links_raw.loc[filt, :]
links_raw_diff = links_raw.loc[filt == False, :]
links_raw_diff.to_csv("data/links_raw_diff_30m_infer.csv", index=False)

links_df = links_raw_diff[["link", "frm_node", "to_node", "dist", "dist_corr", "time"]].groupby(["link", "frm_node", "to_node"], as_index=False).median()
links_nsample = summary(links_raw_diff["link"])
links_nsample.columns = ["n_link_sample"]
links_df = links_df.merge(links_nsample, how="left", left_on="link", right_index=True)

links_df["oppo_link"] = links_df["to_node"] + "-" + links_df["frm_node"]
oppo_df = links_df[["link", "dist_corr", "time", "n_link_sample"]].copy()
oppo_df.columns = ["oppo_link", "oppo_dist", "oppo_time", "oppo_n_link_sample"]
links_df = links_df.merge(oppo_df, how="left", on="oppo_link")
links_df["oppo_n_link_sample"] = links_df["oppo_n_link_sample"].fillna(0).astype(int)
links_df["bidirection"] = pd.isna(links_df["oppo_dist"]) == False
links_df.loc[links_df["bidirection"], "dist_diff"] = np.abs(links_df.loc[links_df["bidirection"], "dist_corr"] - links_df.loc[links_df["bidirection"], "oppo_dist"])
links_df.loc[links_df["bidirection"], "time_diff"] = np.abs(links_df.loc[links_df["bidirection"], "time"] - links_df.loc[links_df["bidirection"], "oppo_time"])
del links_df["oppo_link"]
dist_diff = links_df.loc[links_df["bidirection"], "dist_diff"].values
time_diff = links_df.loc[links_df["bidirection"], "time_diff"].values
# hist(dist_diff, bins=50, range=(0, 1))
# hist(time_diff, bins=50, range=(0, 600))
enhance_filt = links_df["bidirection"] & (links_df["n_link_sample"] < 3) & (links_df["dist_diff"] < 0.3)
oppo_df = links_df.loc[enhance_filt, ["n_link_sample", "oppo_n_link_sample", "dist_corr", "oppo_dist"]]
n_link_sample_sum = oppo_df["n_link_sample"] + oppo_df["oppo_n_link_sample"]
links_df.loc[enhance_filt, "dist_corr"] = oppo_df["dist_corr"] * oppo_df["n_link_sample"] / n_link_sample_sum + oppo_df["oppo_dist"] * oppo_df["oppo_n_link_sample"] / n_link_sample_sum
links_df.loc[enhance_filt, "n_link_sample"] = 3
links_df.to_csv("data/links_df_30m_infer.csv", index=False)


links_df = pd.read_csv("data/links_df_30m_infer.csv")
links_df = pd.read_csv("data/gps_data_2020-09-01_2020-12-31_valid_trip_data_links.csv")
summary(links_df["n_link_sample"] < 3)


def plt_graph(test_point, ew_km_expand=1, ns_km_expand=1, min_link_samples=1):
    inner_grid_filt = (grids[:, 0] < test_point[0]) & (grids[:, 2] > test_point[0]) & (grids[:, 1] < test_point[1]) & (
                grids[:, 3] > test_point[1])
    inner_grid = grids[np.argwhere(inner_grid_filt)[0, 0]]
    outer_grid = inner_grid.copy()
    outer_grid[:2] = outer_grid[:2]
    outer_grid[2:] = outer_grid[2:]
    outer_grid[0] -= unit_deg_1km * ns_km_expand
    outer_grid[2] += unit_deg_1km * ns_km_expand
    outer_grid[1] -= unit_deg_1km * ew_km_expand
    outer_grid[3] += unit_deg_1km * ew_km_expand
    outer_gps_filt = (gps_data["lat"] > outer_grid[0]) & (gps_data["lat"] < outer_grid[2]) & (gps_data["lng"] > outer_grid[1]) & (
                gps_data["lng"] < outer_grid[3])
    outer_gps_data = gps_data.loc[outer_gps_filt, ["lat", "lng", "orient_sim", "dist", "next_dist"]]
    nodes_filt = (nodes_df["lat"] > outer_grid[0]) & (nodes_df["lat"] < outer_grid[2]) & (nodes_df["lng"] > outer_grid[1]) & (
            nodes_df["lng"] < outer_grid[3])
    sub_nodes = nodes_df.loc[nodes_filt, ["node", "lat", "lng"]]
    sub_links = links_df.loc[links_df["frm_node"].isin(sub_nodes["node"]) & links_df["to_node"].isin(sub_nodes["node"]) & (links_df["n_link_sample"] >= min_link_samples), :]

    plot(outer_gps_data["lng"], outer_gps_data["lat"], marker_size=0.005, color="red", style="scatter", x_scale_ticks=[],
         y_scale_ticks=[], show=True)
    sub_nodes.index = sub_nodes["node"]
    for i in range(len(sub_links)):
        lnk = sub_links.iloc[i, :]
        frm_lat_lng = sub_nodes.loc[lnk["frm_node"], ["lat", "lng"]].values
        to_lat_lng = sub_nodes.loc[lnk["to_node"], ["lat", "lng"]].values
        plt_trajectory([frm_lat_lng[0], to_lat_lng[0]], [frm_lat_lng[1], to_lat_lng[1]], mark_od=False, line_width=0.1)


test_pt = (1.34156574971867, 103.69637563289913)
plt_graph(test_pt, min_link_samples=3, ew_km_expand=1.1, ns_km_expand=1.1)


nodes_df = pd.read_csv("data/nodes_df_30m_infer.csv")
nodes_df = pd.read_csv("data/gps_data_2020-09-01_2020-12-31_valid_trip_data_nodes.csv")
def nearest_node(lat, lng, return_dist_diff=False):
    y = nodes_df["lat"].values - lat
    x = nodes_df["lng"].values - lng
    kms = np.round((x ** 2 + y ** 2) ** 0.5 * np.pi * 6371 / 2 / 90, 3)
    ind = np.argmin(kms)
    if return_dist_diff:
        return kms[ind]
    else:
        s = nodes_df.iloc[ind, 0]
        print(s[s.find("@") + 1:])
        print(s)
        return s


nearest_node(1.298257,103.787851)
nearest_node(1.298257,103.787851)


test_data = pd.read_csv("data/test_data.csv")
test_data["from_node"] = [nearest_node(float(frm[:frm.find(",")]), float(frm[frm.find(",")+1:])) for frm in test_data["from_loc"]]
test_data["to_node"] = [nearest_node(float(frm[:frm.find(",")]), float(frm[frm.find(",")+1:])) for frm in test_data["to_loc"]]
from_dist_diff = [nearest_node(float(frm[:frm.find(",")]), float(frm[frm.find(",")+1:]), return_dist_diff=True) for frm in test_data["from_loc"]]
to_dist_diff = [nearest_node(float(frm[:frm.find(",")]), float(frm[frm.find(",")+1:]), return_dist_diff=True) for frm in test_data["to_loc"]]

summary(np.concatenate([from_dist_diff, to_dist_diff]), print_only=True)
test_data.to_csv("data/test_data_2020-09-01_2020-12-31.csv", index=False)




