# docker container rm spark-submit -f
# docker run -it --name spark-submit --network spark-net --volume /home/liming:/home -e HADOOP_USER_NAME=liming -p 4040:4040 mingsqtt/spark_submit:3.0.1 bash

# $SPARK_HOME/bin/pyspark --conf spark.executor.memory=14G --conf spark.executor.cores=6 --master spark://spark-master:7077


from random import random
import numpy as np
import pandas as pd
import logging
from shapely import geometry
from datetime import datetime, date, timedelta
from pyspark.sql.types import *
from pyspark.sql import Window, Row
from pyspark.sql.functions import *
from pyspark.sql import DataFrameStatFunctions as statFunc
from sklearn.cluster import AgglomerativeClustering
import pickle
import itertools


#################################################################################################################################################
######                                                     
######                                      Milestone 2                     
######      In this stage, the data points that constitute good-quality trip sessions are used to derive the nodes and links of the graph.
######      Nodes and Links are collected and saved as csv files in the submit node
######                                                     
#################################################################################################################################################


def cartesian_array(*arrays):
    n_dim = len(arrays)
    iter = itertools.product(*arrays)
    tuples = list(iter)
    df_data = {}
    for dim in range(n_dim):
        col_name = arrays[dim].name if type(arrays[dim]) == pd.Series else "arr_{}".format(dim + 1)
        df_data[col_name] = list()
    col_names = list(df_data.keys())
    for i, tup in enumerate(tuples):
        for dim in range(n_dim):
            df_data[col_names[dim]].append(tup[dim])
    return pd.DataFrame(df_data)



program_start = datetime.now()

schema = StructType([StructField('trip_id', IntegerType(), nullable=False),
					 StructField('driver', IntegerType(), nullable=False),
                     StructField('lat', DoubleType(), nullable=False),
                     StructField('lng', DoubleType(), nullable=False),
                     StructField('ts_utc', TimestampType(), nullable=False),
                     StructField('time_del', FloatType(), nullable=False),
                     StructField('dist', FloatType(), nullable=False),
                     StructField('speed', FloatType(), nullable=False),
                     StructField('orient', FloatType(), nullable=False),
                     StructField('next_dist', FloatType(), nullable=False),
                     StructField('action', IntegerType(), nullable=False)
                     ])


# valid_trip_data_file_path = "hdfs://192.168.2.110:9000/drivergps/gps_data_y_2020-09-01_2020-12-31_valid_trip_data.orc"
valid_trip_data_file_path = "hdfs://192.168.2.110:9000/drivergps/gps_data_2020-09-01_2020-12-31_valid_trip_data.orc"
nodes_output_file_path = "/home/data" + valid_trip_data_file_path[valid_trip_data_file_path.rfind("/"):-4] + "_nodes.csv"
links_output_file_path = "/home/data" + valid_trip_data_file_path[valid_trip_data_file_path.rfind("/"):-4] + "_links.csv"
trip_data = spark.read.option("inferSchema", True).orc(valid_trip_data_file_path)


win_by_trip = Window.partitionBy("trip_id").orderBy("ts_utc")

trip_data = (trip_data
            .withColumn("lat", round(col("lat"), 7))
            .withColumn("lng", round(col("lng"), 7))
            .withColumn("prev_lat", lag("lat", 1).over(win_by_trip))
            .withColumn("prev_lng", lag("lng", 1).over(win_by_trip))
            .withColumn("next_lat", lead("lat", 1).over(win_by_trip))
            .withColumn("next_lng", lead("lng", 1).over(win_by_trip))
            .withColumn("next_dist", lead("dist", 1).over(win_by_trip))
            )
# filling null values for the first data point for each trip
trip_data = (trip_data
            .withColumn("first", col("prev_lat").isNull())
            .withColumn("prev_lat", when(col("first"), col("lat")).otherwise(col("prev_lat")))
            .withColumn("prev_lng", when(col("first"), col("lng")).otherwise(col("prev_lng")))
            )
# filling null values for the last data point for each trip
trip_data = (trip_data
            .withColumn("last", col("next_lat").isNull())
            .withColumn("next_lat", when(col("last"), col("lat")).otherwise(col("next_lat")))
            .withColumn("next_lng", when(col("last"), col("lng")).otherwise(col("next_lng")))
            .withColumn("next_dist", when(col("last"), lit(0)).otherwise(col("next_dist")))
            )

trip_data = (trip_data
            .withColumn("next2_lat", lead("lat", 2).over(win_by_trip))
            .withColumn("next2_lng", lead("lng", 2).over(win_by_trip))
            )
trip_data = (trip_data
            .withColumn("last2", col("last") | col("next2_lat").isNull())
            .withColumn("next2_lat", when(col("last2"), col("next_lat")).otherwise(col("next2_lat")))
            .withColumn("next2_lng", when(col("last2"), col("next_lng")).otherwise(col("next2_lng")))
            )


trip_data = (trip_data
            .withColumn("_y1", when(col("first"), lit(0)).otherwise((col("lat") - col("prev_lat"))))
            .withColumn("_x1", when(col("first"), lit(0)).otherwise((col("lng") - col("prev_lng"))))
            .withColumn("_y2", when(col("last"), lit(0)).otherwise((col("next_lat") - col("lat"))))
            .withColumn("_x2", when(col("last"), lit(0)).otherwise((col("next_lng") - col("lng"))))
            .withColumn("_y3", (col("next2_lat") - col("lat")))
            .withColumn("_x3", (col("next2_lng") - col("lng")))
            .withColumn("_d2", ((col("_y1") ** 2 + col("_x1") ** 2) ** 0.5 * (col("_y2") ** 2 + col("_x2") ** 2) ** 0.5))
            .withColumn("_d3", ((col("_y1") ** 2 + col("_x1") ** 2) ** 0.5 * (col("_y3") ** 2 + col("_x3") ** 2) ** 0.5))
            )

trip_data = (trip_data
            .withColumn("_sim2", when(col("_d2") == 0, lit(1)).otherwise((col("_y1") * col("_y2") + col("_x1") * col("_x2")) / col("_d2")))
            .withColumn("_sim3", when(col("_d3") == 0, lit(1)).otherwise((col("_y1") * col("_y3") + col("_x1") * col("_x3")) / col("_d3")))
            .withColumn("orient_sim", when((col("last2") == False) & (col("next_dist") < 0.03) & (col("_sim3") < col("_sim2")), col("_sim3")).otherwise(col("_sim2")))
            )


cols = ["trip_id", "lat", "lng", "ts_utc", "time_del", "dist", "next_dist", "orient_sim"]
trip_data = trip_data.select(cols).cache()

global_from_lat, global_from_lng = 1.2, 103.6
global_to_lat, global_to_lng = 1.48, 104.1
unit_deg_1km = np.round(((1 * 90 * 2 / 6371 / np.pi) ** 2) ** 0.5, 3)
unit_deg_0p2km = np.round(((0.2 * 90 * 2 / 6371 / np.pi) ** 2) ** 0.5, 3)
dist_thresh = 0.15
clust1_dist_thresh = 0.001
clust2_dist_thresh = 0.001
min_clust_size = 5
nrow = int(np.ceil((global_to_lat - global_from_lat) / unit_deg_1km))
ncol = int(np.ceil((global_to_lng - global_from_lng) / unit_deg_1km))
lats = np.round(np.linspace(global_from_lat, global_from_lat + nrow * unit_deg_1km, nrow + 1), 3)
longs = np.round(np.linspace(global_from_lng, global_from_lng + ncol * unit_deg_1km, ncol + 1), 3)
grids_frm = cartesian_array(lats, longs, return_as_dataframe=True).values
grids = np.concatenate([grids_frm, grids_frm + unit_deg_1km], axis=1)


def map_to_processing_regions(row):
    row_dicts = list()
    ori_row_dict = row.asDict()
    lat = ori_row_dict["lat"]
    lng = ori_row_dict["lng"]
    cell_y = int((lat - global_from_lat) / unit_deg_1km)
    cell_x = int((lng - global_from_lng) / unit_deg_1km)
    ori_row_dict["cell_y"] = cell_y
    ori_row_dict["cell_x"] = cell_x
    ori_row_dict["aug"] = False
    row_dicts.append(ori_row_dict)
    cell_from_lat, cell_to_lat = np.round(global_from_lat + (cell_y * unit_deg_1km), 3), np.round(global_from_lat + ((cell_y + 1) * unit_deg_1km), 3)
    cell_from_lng, cell_to_lng = np.round(global_from_lng + (cell_x * unit_deg_1km), 3), np.round(global_from_lng + ((cell_x + 1) * unit_deg_1km), 3)
    center_from_lat, center_to_lat = cell_from_lat + unit_deg_0p2km, cell_to_lat - unit_deg_0p2km
    center_from_lng, center_to_lng = cell_from_lng + unit_deg_0p2km, cell_to_lng - unit_deg_0p2km
    if (center_from_lat <= lat <= center_to_lat) and (center_from_lng <= lng <= center_to_lng):
        # center
        pass
    elif lng < center_from_lng:
        copy_row_dict = ori_row_dict.copy()
        copy_row_dict["cell_y"] = cell_y
        copy_row_dict["cell_x"] = cell_x - 1
        copy_row_dict["aug"] = True
        row_dicts.append(copy_row_dict)
        if center_from_lat <= lat <= center_to_lat:
            # left
            pass
        elif lat < center_from_lat:
            # bottom-left
            copy_row_dict = ori_row_dict.copy()
            copy_row_dict["cell_y"] = cell_y - 1
            copy_row_dict["cell_x"] = cell_x
            copy_row_dict["aug"] = True
            row_dicts.append(copy_row_dict)
            copy_row_dict = ori_row_dict.copy()
            copy_row_dict["cell_y"] = cell_y - 1
            copy_row_dict["cell_x"] = cell_x - 1
            copy_row_dict["aug"] = True
            row_dicts.append(copy_row_dict)
        else:
            # top-left
            copy_row_dict = ori_row_dict.copy()
            copy_row_dict["cell_y"] = cell_y + 1
            copy_row_dict["cell_x"] = cell_x
            copy_row_dict["aug"] = True
            row_dicts.append(copy_row_dict)
            copy_row_dict = ori_row_dict.copy()
            copy_row_dict["cell_y"] = cell_y + 1
            copy_row_dict["cell_x"] = cell_x - 1
            copy_row_dict["aug"] = True
            row_dicts.append(copy_row_dict)
    elif lng > center_to_lng:
        copy_row_dict = ori_row_dict.copy()
        copy_row_dict["cell_y"] = cell_y
        copy_row_dict["cell_x"] = cell_x + 1
        copy_row_dict["aug"] = True
        row_dicts.append(copy_row_dict)
        if center_from_lat <= lat <= center_to_lat:
            # right
            pass
        elif lat < center_from_lat:
            # bottom-right
            copy_row_dict = ori_row_dict.copy()
            copy_row_dict["cell_y"] = cell_y - 1
            copy_row_dict["cell_x"] = cell_x
            copy_row_dict["aug"] = True
            row_dicts.append(copy_row_dict)
            copy_row_dict = ori_row_dict.copy()
            copy_row_dict["cell_y"] = cell_y - 1
            copy_row_dict["cell_x"] = cell_x + 1
            copy_row_dict["aug"] = True
            row_dicts.append(copy_row_dict)
        else:
            # top-right
            copy_row_dict = ori_row_dict.copy()
            copy_row_dict["cell_y"] = cell_y + 1
            copy_row_dict["cell_x"] = cell_x
            copy_row_dict["aug"] = True
            row_dicts.append(copy_row_dict)
            copy_row_dict = ori_row_dict.copy()
            copy_row_dict["cell_y"] = cell_y + 1
            copy_row_dict["cell_x"] = cell_x + 1
            copy_row_dict["aug"] = True
            row_dicts.append(copy_row_dict)
    elif lat < center_from_lat:
        # bottom
        copy_row_dict = ori_row_dict.copy()
        copy_row_dict["cell_y"] = cell_y - 1
        copy_row_dict["cell_x"] = cell_x
        copy_row_dict["aug"] = True
        row_dicts.append(copy_row_dict)
    else:
        # top
        copy_row_dict = ori_row_dict.copy()
        copy_row_dict["cell_y"] = cell_y + 1
        copy_row_dict["cell_x"] = cell_x
        copy_row_dict["aug"] = True
        row_dicts.append(copy_row_dict)
    return [Row(**row_dict) for row_dict in row_dicts]


augmented_trip_data = sqlContext.createDataFrame(trip_data.rdd.flatMap(map_to_processing_regions))
augmented_trip_data = augmented_trip_data.withColumn("cell_id", concat(col("cell_y"), lit("-"), col("cell_x")))


def make_assign_nodes(cell_rows):
    lat_arr = np.zeros(len(cell_rows))
    lng_arr = np.zeros(len(cell_rows))
    orient_sim_arr = np.zeros(len(cell_rows))
    dist_arr = np.zeros(len(cell_rows))
    row_dicts = [None]*len(cell_rows)
    for r, row in enumerate(cell_rows):
        row_dict = row.asDict()
        lat_arr[r] = row_dict["lat"]
        lng_arr[r] = row_dict["lng"]
        orient_sim_arr[r] = row_dict["orient_sim"]
        dist_arr[r] = row_dict["dist"]
        cell_y = row_dict["cell_y"]
        cell_x = row_dict["cell_x"]
        row_dict["node"] = None
        row_dict["node_lat"] = 0.0
        row_dict["node_lng"] = 0.0
        row_dicts[r] = row_dict
    cell_from_lat, cell_to_lat = np.round(global_from_lat + (cell_y * unit_deg_1km), 3), np.round(global_from_lat + ((cell_y + 1) * unit_deg_1km), 3)
    cell_from_lng, cell_to_lng = np.round(global_from_lng + (cell_x * unit_deg_1km), 3), np.round(global_from_lng + ((cell_x + 1) * unit_deg_1km), 3)
    outer_gps_data = pd.DataFrame({"row_ind": range(len(cell_rows))})
    #
    orient_samples = orient_sim_arr[orient_sim_arr > -0.5]
    if len(orient_samples) > 0:
        orient_sim_thresh = np.minimum(np.quantile(np.abs(orient_samples), 0.1), 0.85)
        conj_filt = (orient_sim_arr > -0.5) & (np.abs(orient_sim_arr) < orient_sim_thresh) & (dist_arr > dist_thresh)
        conj_points = np.stack([lng_arr[conj_filt], lat_arr[conj_filt]], axis=1)
        if len(conj_points) >= min_clust_size:
            clusters = AgglomerativeClustering(n_clusters=None, affinity="euclidean", distance_threshold=clust1_dist_thresh,
                                           linkage="complete").fit(conj_points)
            clust_lv1 = pd.DataFrame(
            {
                "lng": conj_points[:, 0],
                "lat": conj_points[:, 1],
                "c1": clusters.labels_.astype(str)
            })
            outer_gps_data["c1"] = None
            outer_gps_data.loc[conj_filt, "c1"] = clust_lv1["c1"].values
            #
            clust_size = pd.DataFrame({"c1": clust_lv1["c1"], "c1_size": 1}).groupby("c1", as_index=False).sum()
            valid_clust = clust_size.loc[clust_size["c1_size"] >= np.maximum(np.minimum(min_clust_size, np.max(clust_size["c1_size"])), 2), "c1"]
            clust_lv1 = clust_lv1.loc[clust_lv1["c1"].isin(valid_clust), :]
            outer_gps_data.loc[outer_gps_data["c1"].isin(valid_clust) == False, "c1"] = None
            n_lv1_clust = len(valid_clust)
            #
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
            #
            clust_lv2["c_lat"] = np.round(clust_lv2["c_lat"], 6)
            clust_lv2["c_lng"] = np.round(clust_lv2["c_lng"], 6)
            clust_lv2["node"] = None
            node_filt = (clust_lv2["c_lat"] >= cell_from_lat) & (clust_lv2["c_lat"] < cell_to_lat) & (clust_lv2["c_lng"] >= cell_from_lng) & (clust_lv2["c_lng"] < cell_to_lng)
            clust_lv2.loc[node_filt, "node"] = clust_lv2.loc[node_filt, "c2"] + "@" + clust_lv2.loc[node_filt, "c_lat"].astype(str) + "," + clust_lv2.loc[node_filt, "c_lng"].astype(str)
            outer_gps_data = outer_gps_data.merge(clust_lv2, how="left", on="c2")
            for i, row in outer_gps_data.loc[pd.isna(outer_gps_data["node"]) == False, ["row_ind", "node", "c_lat", "c_lng"]].iterrows():
                row_dict = row_dicts[row["row_ind"]]
                row_dict["node"] = row["node"]
                row_dict["node_lat"] = row["c_lat"]
                row_dict["node_lng"] = row["c_lng"]
            node_dicts = list()
            for i, row in clust_lv2.loc[node_filt, :].iterrows():
                node_dicts.append({
                    "node": row["node"],
        			"clust_size": row["clust_size"],
        			"lat": row["c_lat"],
        			"lng": row["c_lng"]
        			})
            return ([Row(**row_dict) for row_dict in row_dicts if not row_dict["aug"]], [Row(**node_dict) for node_dict in node_dicts], (len(cell_rows), len(conj_points), n_lv1_clust, len(clust_lv2), np.sum(node_filt)))
        else:
            return ([Row(**row_dict) for row_dict in row_dicts if not row_dict["aug"]], [], (len(cell_rows), len(conj_points), 0, 0, 0))
    else:
        return ([Row(**row_dict) for row_dict in row_dicts if not row_dict["aug"]], [], (len(cell_rows), 0, 0, 0, 0))


# map all rows to PairRDD of (cell_id, Row)
pairs = augmented_trip_data.rdd.map(lambda row: (row["cell_id"], row))

# group the PairRDD by cell_id as (cell_id, Iterable<Row>)
# the output of the mapValue operation is (cell_id, (List<Row>, List<Row>, stats_tuple)
tuples_by_cell = pairs.groupByKey().mapValues(make_assign_nodes).cache()

# the output of the flatMapValues operation is PairRDD of (cell_id, Row)
assigned_rows_flatten = tuples_by_cell.flatMapValues(lambda tup: tup[0])
assigned_trip_data = sqlContext.createDataFrame(assigned_rows_flatten.values())

# the output of the flatMapValues operation is PairRDD of (cell_id, Row)
node_rows_flatten = tuples_by_cell.flatMapValues(lambda tup: tup[1])
nodes_df = sqlContext.createDataFrame(node_rows_flatten.values()).toPandas()


def associate_nearby_nodes(trip_rows):
    lat_arr = np.zeros(len(trip_rows))
    lng_arr = np.zeros(len(trip_rows))
    row_dicts = [None]*len(trip_rows)
    for r, row in enumerate(trip_rows):
        row_dict = row.asDict()
        lat_arr[r] = row_dict["lat"]
        lng_arr[r] = row_dict["lng"]
        row_dict["inferred_node"] = None
        row_dict["inferred_node_lat"] = 0.0
        row_dict["inferred_node_lng"] = 0.0
        row_dicts[r] = row_dict
    mtx_node2pt_lat = np.repeat(nodes_df["lat"].values, len(trip_rows)).reshape((-1, len(trip_rows)))
    mtx_node2pt_lng = np.repeat(nodes_df["lng"].values, len(trip_rows)).reshape((-1, len(trip_rows)))
    mtx_pt2node_lat = np.repeat(lat_arr.reshape((-1, 1)), len(nodes_df), axis=1).transpose()
    mtx_pt2node_lng = np.repeat(lng_arr.reshape((-1, 1)), len(nodes_df), axis=1).transpose()
    y = mtx_node2pt_lat - mtx_pt2node_lat
    x = mtx_node2pt_lng - mtx_pt2node_lng
    mtx_km = np.round((x ** 2 + y ** 2) ** 0.5 * np.pi * 6371 / 2 / 90, 3)
    nearest_node_indices = np.argmin(mtx_km, axis=0)
    shortest_dists = mtx_km[nearest_node_indices, range(len(nearest_node_indices))]
    matched_pt_indices = np.argwhere(shortest_dists < 0.03).flatten()
    if len(matched_pt_indices) > 0:
        nearby_node_info = nodes_df.loc[nearest_node_indices[matched_pt_indices], ["node", "lat", "lng"]].values
        for k, r in enumerate(matched_pt_indices):
            row_dicts[r]["inferred_node"] = nearby_node_info[k, 0]
            row_dicts[r]["inferred_node_lat"] = nearby_node_info[k, 1]
            row_dicts[r]["inferred_node_lng"] = nearby_node_info[k, 2]
    return [Row(**row_dict) for row_dict in row_dicts]


# map all rows to PairRDD of (trip_id, Row)
pairs = assigned_trip_data.rdd.map(lambda row: (row["trip_id"], row))

# group the PairRDD by trip_id as (trip_id, Iterable<Row>)
# the output of the mapValue operation is (trip_id, List<Row>)
rows_by_trip = pairs.groupByKey().mapValues(associate_nearby_nodes)

# the output of the flatMapValues operation is PairRDD of (trip_id, Row)
inferred_rows_flatten = rows_by_trip.flatMapValues(lambda rows: rows)
assigned_trip_data = sqlContext.createDataFrame(inferred_rows_flatten.values())
assigned_trip_data = (assigned_trip_data
                    .withColumn("eff_node", when(col("node").isNull(), col("inferred_node")).otherwise(col("node")))
                    .withColumn("eff_node_lat", when(col("node").isNull(), col("inferred_node_lat")).otherwise(col("node_lat")))
                    .withColumn("eff_node_lng", when(col("node").isNull(), col("inferred_node_lng")).otherwise(col("node_lng")))
                    .withColumn("dist_corr", when(col("time_del") > 3*60, col("dist") * 1.4).otherwise(col("dist") * 1.1))
                    .repartition("trip_id")
                    .cache()
                    )

n_nodes = assigned_trip_data.select("trip_id", "eff_node").groupBy("trip_id").agg(count("eff_node").alias("n_node"))
multinode_trip_data = assigned_trip_data.join(n_nodes, ["trip_id"], how="inner").filter(col("n_node") > 1)


def make_links(trip_rows):
    lat_arr = np.zeros(len(trip_rows))
    lng_arr = np.zeros(len(trip_rows))
    dist_arr = np.zeros(len(trip_rows))
    dist_corr_arr = np.zeros(len(trip_rows))
    time_del_arr = np.zeros(len(trip_rows))
    node_arr = np.array([None]*len(trip_rows))
    unsorted_row_dicts = [row.asDict() for row in trip_rows]
    for r, row_dict in enumerate(sorted(unsorted_row_dicts, key = lambda rd: rd["ts_utc"])):
        lat_arr[r] = row_dict["lat"]
        lng_arr[r] = row_dict["lng"]
        dist_arr[r] = row_dict["dist"]
        dist_corr_arr[r] = row_dict["dist_corr"]
        time_del_arr[r] = row_dict["time_del"]
        node_arr[r] = row_dict["eff_node"]
        trip_id = row_dict["trip_id"]
    node_indices = np.argwhere(pd.isna(node_arr) == False).flatten()
    link_row_dicts = list()
    for n in range(len(node_indices) - 1):
        frm = node_indices[n]
        to = node_indices[n + 1]
        frm_lat, frm_lng, frm_node = lat_arr[frm], lng_arr[frm], node_arr[frm]
        to_lat, to_lng, to_node = lat_arr[to], lng_arr[to], node_arr[to]
        if frm_node == to_node:
            continue
        dist_sum = np.sum(dist_arr[frm + 1:to + 1])
        time_sum = np.sum(time_del_arr[frm + 1:to + 1])
        dist_corr_sum = np.sum(dist_corr_arr[frm + 1:to + 1])
        link_row_dicts.append({
            "link": "{}-{}".format(frm_node, to_node),
            "frm_pt_lat": float(frm_lat),
            "frm_pt_lng": float(frm_lng),
            "frm_node": frm_node,
            "to_pt_lat": float(to_lat),
            "to_pt_lng": float(to_lng),
            "to_node": to_node,
            "dist": float(dist_sum),
            "dist_corr": float(dist_corr_sum),
            "time": float(time_sum),
            "trip_id": trip_id,
        })
    return [Row(**row_dict) for row_dict in link_row_dicts]
        


# map all rows to PairRDD of (trip_id, Row)
pairs = multinode_trip_data.rdd.map(lambda row: (row["trip_id"], row))

# group the PairRDD by trip_id as (trip_id, Iterable<Row>)
# the output of the mapValue operation is (trip_id, List<Row>)
rows_by_trip = pairs.groupByKey().mapValues(make_links)


# the output of the flatMapValues operation is PairRDD of (trip_id, Row)
rows_flatten = rows_by_trip.flatMapValues(lambda rows: rows)
raw_links_data = sqlContext.createDataFrame(rows_flatten.values())


def calc_link_stats(link_rows):
    dist_corr_arr = np.zeros(len(link_rows))
    for r, row_dict in enumerate([row.asDict() for row in link_rows]):
        dist_corr_arr[r] = row_dict["dist_corr"]
        link = row_dict["link"]
        frm_node = row_dict["frm_node"]
        to_node = row_dict["to_node"]
    return Row(**{
        "link": link, 
        "frm_node": frm_node, 
        "to_node": to_node, 
        "dist_corr": float(np.round(np.median(dist_corr_arr), 3)),
        "n_link_sample": len(link_rows)
        })


# map all rows to PairRDD of (link, Row)
pairs = raw_links_data.rdd.map(lambda row: (row["link"], row))

# group the PairRDD by link as (link, Iterable<Row>)
# the output of the mapValue operation is (link, Row)
# materialize links_df to Pandas dataframe because it will have relative small size
links_df = sqlContext.createDataFrame(pairs.groupByKey().mapValues(calc_link_stats).values()).toPandas()


links_df["oppo_link"] = links_df["to_node"] + "-" + links_df["frm_node"]
oppo_df = links_df[["link", "dist_corr", "n_link_sample"]].copy()
oppo_df.columns = ["oppo_link", "oppo_dist", "oppo_n_link_sample"]
links_df = links_df.merge(oppo_df, how="left", on="oppo_link")
links_df["oppo_n_link_sample"] = links_df["oppo_n_link_sample"].fillna(0).astype(int)
links_df["bidirection"] = pd.isna(links_df["oppo_dist"]) == False
links_df.loc[links_df["bidirection"], "dist_diff"] = np.abs(links_df.loc[links_df["bidirection"], "dist_corr"] - links_df.loc[links_df["bidirection"], "oppo_dist"])
del links_df["oppo_link"]
dist_diff = links_df.loc[links_df["bidirection"], "dist_diff"].values
enhance_filt = links_df["bidirection"] & (links_df["n_link_sample"] < 3) & (links_df["dist_diff"] < 0.3)
oppo_df = links_df.loc[enhance_filt, ["n_link_sample", "oppo_n_link_sample", "dist_corr", "oppo_dist"]]
n_link_sample_sum = oppo_df["n_link_sample"] + oppo_df["oppo_n_link_sample"]
links_df.loc[enhance_filt, "dist_corr"] = oppo_df["dist_corr"] * oppo_df["n_link_sample"] / n_link_sample_sum + oppo_df["oppo_dist"] * oppo_df["oppo_n_link_sample"] / n_link_sample_sum
links_df.loc[enhance_filt, "n_link_sample"] = 3

nodes_df.to_csv(nodes_output_file_path, index=False)
links_df.to_csv(links_output_file_path, index=False)


