# docker container rm spark-submit -f
# docker run -it --name spark-submit --network spark-net --volume /home/liming:/home -e HADOOP_USER_NAME=liming -p 4040:4040 mingsqtt/spark_submit:3.0.1 bash

# $SPARK_HOME/bin/pyspark --conf spark.executor.memory=14G --conf spark.executor.cores=6 --master spark://spark-master:7077


from random import random
import numpy as np
import logging
from shapely import geometry
from datetime import datetime, date, timedelta
from pyspark.sql.types import *
from pyspark.sql import Window, Row
from pyspark.sql.functions import *
from sklearn.ensemble import RandomForestClassifier
import pickle


#################################################################################################################################################
######                                                     
######                                      Milestone 1                       
######      In this stage, raw driver location data is retrieved from an HDFS storage in csv format.
######      Each raw data point is feature-engineered and classifed as either in a "moving" state or in "still" state.
######      The state of the data points is used to infer a collection of candidate trip sessions.
######      Each trip session comprises a series of data points which are in moving state.
######      The candidate trip sessions are examined, such that only reasonable data points are shortlisted and used for graph creation task.
######                                                     
#################################################################################################################################################

program_start = datetime.now()

schema = StructType([StructField('com', IntegerType(), nullable=False),
                     StructField('driver', IntegerType(), nullable=False),
                     StructField('veh', StringType(), nullable=False),
                     StructField('ts_utc', TimestampType(), nullable=False),
                     StructField('lat', DoubleType(), nullable=False),
                     StructField('lng', DoubleType(), nullable=False),
                     ])

# schema = StructType([StructField('driver', IntegerType(), nullable=False),
#                      StructField('ts_utc', TimestampType(), nullable=False),
#                      StructField('lat', DoubleType(), nullable=False),
#                      StructField('lng', DoubleType(), nullable=False),
#                      ])

# defining window for window operation
# a window is sliding along the timeline for each driver. data points pertaining to different drivers are independent.
win_by_driver = Window.partitionBy("driver").orderBy("ts_utc")
c3_win_by_driver = Window.partitionBy("driver").orderBy("ts_utc").rowsBetween(-1, 1)

input_file_path = "hdfs://192.168.2.110:9000/drivergps/gps_data_2020-09-01_2020-12-31.csv"
# input_file_path = "hdfs://192.168.2.110:9000/drivergps/gps_data_x_2020-12-01_2020-12-31_com479.csv"
stage1_file_path = input_file_path[:-4] + "_stage1.orc"
stage2_file_path = input_file_path[:-4] + "_stage2.orc"
trip_session_stats_file_path = input_file_path[:-4] + "_candi_trip_sessions.orc"
valid_sessions_stats_file_path = input_file_path[:-4] + "_valid_trip_sessions.orc"
valid_trip_data_file_path = input_file_path[:-4] + "_valid_trip_data.orc"

# reading raw data points
gps_data = spark.read.csv(input_file_path, schema=schema, header=True)
total_source_data_count = gps_data.count()

# make sure the data points are in chronological order
gps_data = gps_data.orderBy("driver", "ts_utc")

##
## calculate time delta between data points in order to remove high-freq data points  (known mobile app bug)
##

# get each data point's previous time lag's timestamp
gps_data = gps_data.withColumn("prev_ts", lag("ts_utc", 1).over(win_by_driver))
# for each driver's first data point, set prev_ts = ts_utc for filling null value
gps_data = gps_data.withColumn("prev_ts", when(col("prev_ts").isNull(), col("ts_utc")).otherwise(col("prev_ts")))
# calc the time detal (in sec), between a data point and its preceding data point
gps_data = gps_data.withColumn("time_del", round(col("ts_utc").cast(DoubleType()) - col("prev_ts").cast(DoubleType()), 1))
#
# set the 1st time_del/dist/orient to be same as the 2nd time_del/dist/orient for each driver
# this is based on the assumption that a driver's first data point retains the same moving/still state from its previous data point which is not part of the current dataset
#
gps_data = gps_data.withColumn("next_time_del", lead("time_del", 1).over(win_by_driver))
gps_data = gps_data.withColumn("time_del", when((col("ts_utc") == col("prev_ts")) & (col("next_time_del").isNull() == False), col("next_time_del")).otherwise(col("time_del")))
# discard data points that are sampled at too-high frequency. 
# the expected sampling frequency is once per 60sec, i.e. time_del ~ 60
gps_data = gps_data.filter(col("time_del") > 10)
# gps_data.select("driver", "time_del").filter((col("driver") == 1052)).groupBy("driver").agg(sum("time_del")).show(300)
##
## feature engineering for data point state classification task
##

# get prev/next timestamp, prev/next lat/lng
gps_data = (gps_data.withColumn("prev_ts", lag("ts_utc", 1).over(win_by_driver))
            .withColumn("prev_lat", lag("lat", 1).over(win_by_driver))
            .withColumn("prev_lng", lag("lng", 1).over(win_by_driver))
            .withColumn("next_ts", lead("ts_utc", 1).over(win_by_driver))
            .withColumn("next_lat", lead("lat", 1).over(win_by_driver))
            .withColumn("next_lng", lead("lng", 1).over(win_by_driver))
            )
# filling null values for the first data point for each driver
gps_data = (gps_data
            .withColumn("first", col("prev_ts").isNull())
            .withColumn("prev_ts", when(col("first"), col("ts_utc")).otherwise(col("prev_ts")))
            .withColumn("prev_lat", when(col("first"), col("lat")).otherwise(col("prev_lat")))
            .withColumn("prev_lng", when(col("first"), col("lng")).otherwise(col("prev_lng")))
            )
# filling null values for the last data point for each driver
gps_data = (gps_data
            .withColumn("last", col("next_ts").isNull())
            .withColumn("next_ts", when(col("last"), col("ts_utc")).otherwise(col("next_ts")))
            .withColumn("next_lat", when(col("last"), col("lat")).otherwise(col("next_lat")))
            .withColumn("next_lng", when(col("last"), col("lng")).otherwise(col("next_lng")))
            )
# calc the time delta from the preceding/succeeding data point to the target data point
gps_data = (gps_data
            .withColumn("time_del", round(col("ts_utc").cast(DoubleType()) - col("prev_ts").cast(DoubleType()), 1))
            .withColumn("time_to_next_lag", round(col("next_ts").cast(DoubleType()) - col("ts_utc").cast(DoubleType()), 1))
            )
# calc Euclidean distance (in KM) and angle (vehicle's orientation) from the preceding data point to the target data point
gps_data = (gps_data
            .withColumn("_y", (col("lat") - col("prev_lat")))
            .withColumn("_x", (col("lng") - col("prev_lng")))
            )
gps_data = gps_data.withColumn("dist", round((col("_x")**2 + col("_y")**2) ** 0.5 * np.pi * 6371 / 2 / 90, 3))
gps_data = gps_data.withColumn("orient", round(atan2(col("_y"), col("_x")), 3))
# get distance and orientation from the target data point to the succeeding data point
gps_data = (gps_data
            .withColumn("next_dist", lead("dist", 1).over(win_by_driver))
            .withColumn("next_orient", lead("orient", 1).over(win_by_driver))
            )
gps_data = (gps_data
            .withColumn("next_dist", when(col("last"), col("dist")).otherwise(col("next_dist")))
            .withColumn("next_orient", when(col("last"), col("orient")).otherwise(col("next_orient")))
            )
#
# set the 1st time_del/dist/orient to be same as the 2nd time_del/dist/orient for each driver
# this is based on the assumption that a driver's first data point retains the same moving/still state from its previous data point which is not part of the current dataset
#
gps_data = gps_data.withColumn("next_time_del", lead("time_del", 1).over(win_by_driver))
gps_data = (gps_data
            .withColumn("time_del", when(col("first"), col("next_time_del")).otherwise(col("time_del")))
            .withColumn("dist", when(col("first"), col("next_dist")).otherwise(col("dist")))
            .withColumn("orient", when(col("first"), col("next_orient")).otherwise(col("orient")))
            )
# get distance and orientation from the 2nd preceding data point to the 1st preceding data point
gps_data = (gps_data
            .withColumn("prev_dist", lag("dist", 1).over(win_by_driver))
            .withColumn("prev_orient", lag("orient", 1).over(win_by_driver))
            )
gps_data = (gps_data
            .withColumn("prev_dist", when(col("first"), col("dist")).otherwise(col("prev_dist")))
            .withColumn("prev_orient", when(col("first"), col("orient")).otherwise(col("prev_orient")))
            )
# get distance and orientation from the 3rd preceding data point to the 2nd preceding data point
gps_data = gps_data.withColumn("prev2_dist", lag("dist", 2).over(win_by_driver))
gps_data = gps_data.withColumn("first2", col("prev2_dist").isNull() | col("first"))
gps_data = gps_data.withColumn("prev2_dist", when(col("first2"), col("prev_dist")).otherwise(col("prev2_dist")))
# calc the slope of change from prev_dist to dist
gps_data = gps_data.withColumn("dist_slope", round(col("dist") / (col("prev_dist") + 1e-3), 4))
# calc speed in KM/Hour
gps_data = gps_data.withColumn("speed", when(col("time_del") > 1, col("dist") / (col("time_del")/3600)).otherwise(col("dist") / (1/3600)))
# get preceding speeds and succeeding speeds
gps_data = gps_data.withColumn("prev_speed", lag("speed", 1).over(win_by_driver)) 
gps_data = gps_data.withColumn("prev_speed", when(col("first"), col("speed")).otherwise(col("prev_speed")))
gps_data = gps_data.withColumn("prev2_speed", lag("speed", 2).over(win_by_driver))
gps_data = gps_data.withColumn("prev2_speed", when(col("first2"), col("prev_speed")).otherwise(col("prev2_speed")))
gps_data = gps_data.withColumn("next_speed", lead("speed", 1).over(win_by_driver))
gps_data = gps_data.withColumn("next_speed", when(col("last"), col("speed")).otherwise(col("next_speed")))
# calc slope of change of speed
gps_data = gps_data.withColumn("speed_slope", round(col("speed") / (col("prev_speed") + 1e-1), 4))
gps_data = gps_data.withColumn("speed_slope2", round(col("speed") / (col("prev2_speed") + 1e-1), 4))
# rouding
gps_data = (gps_data
            .withColumn("speed", round(col("speed"), 1))
            .withColumn("prev_speed", round(col("prev_speed"), 1))
            .withColumn("prev2_speed", round(col("prev2_speed"), 1))
            .withColumn("next_speed", round(col("next_speed"), 1))
            )
#
# set the last time_del/dist/orient to be same as the 2nd last time_del/dist/orient for each driver
# this is based on the assumption that a driver's first data point retains the same moving/still state from its previous data point which is not part of the current dataset
#
gps_data = gps_data.withColumn("prev_time_to_next_lag", lag("time_to_next_lag", 1).over(win_by_driver))
gps_data = gps_data.withColumn("time_to_next_lag", when(col("last"), col("prev_time_to_next_lag")).otherwise(col("time_to_next_lag")))
# discarding temporary columns
# gps_data.select("driver", "time_del", "dist", "orient", "prev_dist", "prev2_dist", "speed", "prev2_speed", "next_speed", "speed_slope2", "time_to_next_lag").filter((col("driver") == 1052)).groupBy("driver").agg(sum("time_del"), sum("dist"), sum("orient"), sum("prev_dist"), sum("prev2_dist"), sum("speed"), sum("prev2_speed"), sum("next_speed"), sum("speed_slope2"), sum("time_to_next_lag")).show(300)
gps_data = gps_data.select("driver", "lat", "lng", "first", "last", "ts_utc", "time_del", "dist", "speed", "prev_dist", "prev2_dist", "prev_speed", "prev2_speed", "orient", "prev_orient", "dist_slope", "speed_slope", "speed_slope2", "time_to_next_lag", "next_speed", "next_dist")


# gps_data.select("time_del", "dist", "speed").agg(sum("time_del"), sum("dist"), sum("speed")).show()

##
## data point state classification
##

# initiate default status value using business rules
gps_data = gps_data.withColumn("status", lit(""))
gps_data = gps_data.withColumn("status", when(((col("speed") > 10) & (col("prev_speed") > 9)) | (col("speed") >= 10), lit("moving")).otherwise(col("status")))
gps_data = gps_data.withColumn("status", when((col("speed") < 3) & (col("prev_speed") < 1.5), lit("still")).otherwise(col("status")))

# load pre-trained RandomForest model
with open("/home/data/truck_status_forest_classifier.pickle", "rb") as f:
    clf = pickle.load(f)


# Map operation
# maps a list of Row object to a new list of Row object of the same length.
# the input Row list pertains to a particular driver.
# state classification will be performed for all the Row objects.
# each new Row object will have a new "eff_status" column which will be used for subsequent processing.
def batch_infer_status(driver_rows):
    # feature columns
    feat_cols = ["dist", "speed", "prev_dist", "prev2_dist", "prev_speed", "prev2_speed", "speed_slope", "speed_slope2", "time_to_next_lag", "next_speed", "next_dist"]
    # prepare input ndarray for classification for all rows
    inputs = np.zeros((len(driver_rows), len(feat_cols)))
    row_dicts = [None]*len(driver_rows)
    for r, row in enumerate(driver_rows):
        row_dicts[r] = row.asDict()
        for c, col in enumerate(feat_cols):
            inputs[r, c] = row[col]
    # RandomForest prediction
    predicts = clf.predict(inputs)
    # derive eff_status value
    for r in range(len(driver_rows)):
        row_dict = row_dicts[r]
        row_dict['inferred_change'] = predicts[r]
        # if "status" value is preset by business rule, use it directly
        if row_dict['status'] != "":
            row_dict['eff_status'] = row_dict['status']
        # if the data point is the first data point for the driver
        elif row_dict['first']:
            if row_dict['speed'] > 5:
                row_dict['eff_status'] = "moving"
            else:
                row_dict['eff_status'] = "still"
        # if the predicted value is "moved"
        elif predicts[r] == "moved":
            row_dict['eff_status'] = "moving"
        # if the predicted value is "stopped"
        elif predicts[r] == "stopped":
            row_dict['eff_status'] = "still"
        # use the "eff_status" value of preceding data point
        else:
            row_dict['eff_status'] = last_status
        last_status = row_dict['eff_status']
    return [Row(**row_dict) for row_dict in row_dicts]


# map all rows to PairRDD of (driver, Row)
pairs = gps_data.rdd.map(lambda row: (row["driver"], row))

# group the PairRDD by driver as (driver, Iterable<Row>), then do status inferrence for each Iterable<Row>
# the output of the mapValue operation is (driver, List<Row>)
rows_by_driver = pairs.groupByKey().mapValues(batch_infer_status)

# flatten the List<Row> to become PairRDD of (driver, Row)
rows_flatten = rows_by_driver.flatMapValues(lambda rows: tuple(rows))

# take all the values of Row and create a new DF
gps_data = sqlContext.createDataFrame(rows_flatten.values())

# checkpoint saving
gps_data.write.format("orc").mode('overwrite').save(stage1_file_path)
del gps_data
gps_data = spark.read.option("inferSchema", True).orc(stage1_file_path)





# some constant value
STOP = 1            # indicates that the vehicle has just come to a stop at the timestamp of a data point
START = 2           # indicates that the vehicle has just started off at the timestamp of a data point
# STOP_SESS_END = 3 
# START_STOP = 4

gps_data = gps_data.withColumn("next_eff_status", lead("eff_status", 1).over(win_by_driver))
gps_data = gps_data.withColumn("next_eff_status", when(col("next_eff_status").isNull(), col("eff_status")).otherwise(col("next_eff_status")))

# move_begin indicates that the vehicle is changing from still state to moving state
# stop_begin indicates that the vehicle is changing from moving state to still state
gps_data = gps_data.withColumn("move_begin", (col("last") == False) & (col("eff_status") == "still") & (col("next_eff_status") == "moving"))
gps_data = gps_data.withColumn("stop_begin", (col("last") == False) & (col("eff_status") == "moving") & (col("next_eff_status") == "still"))
gps_data = gps_data.withColumn("move_end", col("stop_begin"))
gps_data = gps_data.withColumn("stop_end", col("move_begin"))

gps_data = gps_data.withColumn("action", lit(0))
gps_data = gps_data.withColumn("action", when(col("stop_begin") & (col("move_begin") == False), lit(STOP)).otherwise(col("action")))
gps_data = gps_data.withColumn("action", when(col("move_begin") & (col("stop_begin") == False), lit(START)).otherwise(col("action")))
# gps_data = gps_data.withColumn("action", when(col("stop_end") & (col("move_begin") == False), lit(STOP_SESS_END)).otherwise(col("action")))
# gps_data = gps_data.withColumn("action", when(col("move_begin") & col("stop_begin"), lit(START_STOP)).otherwise(col("action")))


# Map operation
# maps a list of Row object to a new list of Row object representing candidate trip session.
# some smoothing work will be done during this operation.
def assign_temp_session_id(driver_rows):
    dists = np.zeros(len(driver_rows))
    speeds = np.zeros(len(driver_rows))
    time_dels = np.zeros(len(driver_rows))
    lats = np.zeros(len(driver_rows))
    lngs = np.zeros(len(driver_rows))
    move_begin_sm = np.zeros(len(driver_rows), dtype=bool)
    move_end_sm = np.zeros(len(driver_rows), dtype=bool)
    stop_begin_sm = np.zeros(len(driver_rows), dtype=bool)
    stop_end_sm = np.zeros(len(driver_rows), dtype=bool)
    actions = np.zeros(len(driver_rows), dtype=int)
    row_dicts = [None]*len(driver_rows)
    for r, row in enumerate(driver_rows):
        row_dict = row.asDict()
        row_dicts[r] = row_dict
        dists[r] = row_dict["dist"]
        speeds[r] = row_dict["speed"]
        time_dels[r] = row_dict["time_del"]
        lats[r] = row_dict["lat"]
        lngs[r] = row_dict["lng"]
        move_begin_sm[r] = row_dict["move_begin"]
        move_end_sm[r] = row_dict["move_end"]
        stop_begin_sm[r] = row_dict["stop_begin"]
        stop_end_sm[r] = row_dict["stop_end"]
        actions[r] = row_dict["action"]
        row_dict["temp_sess"] = 0
        row_dict["temp_sess_first"] = False
    action_indices = np.argwhere(actions != 0).flatten()
    #
    if len(action_indices) >= 2:
        i = 1
        while i < len(action_indices):
            ind1 = action_indices[i-1]
            ind2 = action_indices[i]
            evt1 = actions[ind1]
            evt2 = actions[ind2]
            # if vehicle has travelled for some distance and come to a stop
            if (evt1 == START) and (evt2 == STOP):
                travel_dist = np.sum(dists[(ind1+1):(ind2+1)])
                mean_spd = np.mean(speeds[(ind1+1):(ind2+1)])
                sd_lat = np.std(lats[ind1:(ind2+1)])
                sd_lng = np.std(lngs[ind1:(ind2+1)])
                # if the travel distance is > 0.3KM and average speed is > 5 KMh and the movement is not hovering within a small circle
                if (travel_dist > 0.3) and (mean_spd >= 5) and ((sd_lat > 0.001) or (sd_lng > 0.001)):
                    pass
                # else, invalidate the preceding START event and current STOP event, consider the vehicle is never moved.
                elif i > 1:
                    action_indices = np.delete(action_indices, i)
                    action_indices = np.delete(action_indices, i - 1)
                    i -= 2
                    #
                    move_begin_sm[ind1] = False
                    stop_end_sm[ind1] = False
                    move_end_sm[ind2] = False
                    stop_begin_sm[ind2] = False
            # if vehicle has been still for a while and gets starting off
            elif (evt1 == STOP) and (evt2 == START):
                slow_time = np.sum(time_dels[(ind1+1):(ind2+1)])
                # if the duration of being still is > 120s
                if slow_time > 120:
                    pass
                # else, invalidate the preceding STOP event and current START event, consider the vehicle is always on move and never stopped
                elif i > 1:
                    action_indices = np.delete(action_indices, i)
                    action_indices = np.delete(action_indices, i - 1)
                    i -= 2
                    #
                    stop_begin_sm[ind1] = False
                    move_end_sm[ind1] = False
                    stop_end_sm[ind2] = False
                    move_begin_sm[ind2] = False
            # if two consecutive START events or STOP events appear, invalidate one of them
            elif evt1 == evt2:
                action_indices = np.delete(action_indices, i)
                i -= 1
            i += 1
    # get smoothened actions
    action_sm = np.zeros(len(driver_rows), dtype=int)
    action_sm[stop_begin_sm & (move_begin_sm == False)] = STOP
    action_sm[move_begin_sm & (stop_begin_sm == False)] = START
    # action_sm[stop_end_sm & (move_begin_sm == False)] = STOP_SESS_END
    # action_sm[move_begin_sm & stop_begin_sm] = START_STOP
    #
    action_indices = np.argwhere(action_sm != 0).flatten()
    engine_actions = action_sm[action_indices]
    if len(action_indices) > 0:
        move_begin_indices = action_indices[(engine_actions == START) & (action_indices != np.max(action_indices))]
        move_end_indices = action_indices[(engine_actions == STOP) & (action_indices != np.min(action_indices))]
    else:
        move_begin_indices = np.zeros(0)
        move_end_indices = np.zeros(0)
    assert len(move_begin_indices) == len(move_end_indices), "move_begin_indices and move_end_indices don't have the same length"
    # assign a temp trip session id to data points that make up a trip
    for i in range(len(move_begin_indices)):
        from_ind = (move_begin_indices[i])
        to_ind = (move_end_indices[i]+1)
        for r, row in enumerate(row_dicts[from_ind:to_ind]):
            row["temp_sess"] = row["driver"] * 100000 + i
            row["temp_sess_first"] = (r == 0)
    return [Row(**row_dict) for row_dict in row_dicts]


# map all rows to PairRDD of (driver, Row)
pairs = gps_data.rdd.map(lambda row: (row["driver"], row))

# group the PairRDD by driver as (driver, Iterable<Row>), then assign a temp trip session id to each row in Iterable<Row>
# the output of the mapValue operation is (driver, List<Row>)
rows_by_driver = pairs.groupByKey().mapValues(assign_temp_session_id)

# flatten the List<Row> to become PairRDD of (driver, Row)
rows_flatten = rows_by_driver.flatMapValues(lambda rows: tuple(rows))

# take all the values of Row and create a new DF
gps_data = sqlContext.createDataFrame(rows_flatten.values())

gps_data.write.format("orc").mode('overwrite').save(stage2_file_path)
del gps_data
gps_data = spark.read.option("inferSchema", True).orc(stage2_file_path)


trip_data = gps_data.filter((col("temp_sess") != 0) & (col("temp_sess_first") == False)).cache()
trip_data_lt120 = trip_data.filter(col("speed") < 120)
trip_data_gt40 = trip_data.filter((col("speed") < 120) & (col("speed") > 40))

travel_dist = trip_data.groupBy(["driver", "temp_sess"]).agg(sum("dist").alias("dist")).filter(col("dist") < 120)
travel_time = trip_data.groupBy("temp_sess").agg(
                sum(col("time_del") / 60).alias("dur_mins0"),
                round(sum(col("time_del") / 60), 1).alias("dur_mins"),
                count("time_del").alias("n_sample")
                )
travel_time = (travel_time
                .withColumn("n_sample_ideal", when(round(col("dur_mins0"), 0) > 1, round(col("dur_mins0"), 0)).otherwise(1))
                .withColumn("n_sample_quality", round(col("n_sample") / col("n_sample_ideal"), 2))
              )
ori_max_speed = trip_data.groupBy("temp_sess").agg(max("speed").alias("ori_max_spd"))
speed_mean = trip_data_lt120.groupBy("temp_sess").agg(round(avg("speed"), 1).alias("spd_mean"))
high_speed_count = trip_data_gt40.groupBy("temp_sess").agg(count("speed").alias("high_spd_count"))
high_speed_mean = trip_data_gt40.groupBy("temp_sess").agg(round(avg("speed"), 1).alias("high_spd_mean"))
high_speed_max = trip_data_gt40.groupBy("temp_sess").agg(max("speed").alias("high_spd_max"))


trip_session_stats = (travel_dist
                        .join(travel_time, ["temp_sess"], how="left")
                        .join(ori_max_speed, ["temp_sess"], how="left")
                        .join(speed_mean, ["temp_sess"], how="left")
                        .join(high_speed_count, ["temp_sess"], how="left")
                        .join(high_speed_mean, ["temp_sess"], how="left")
                        .join(high_speed_max, ["temp_sess"], how="left")
                        )
trip_session_stats = (trip_session_stats
                    .withColumn("spd_mean", when(col("spd_mean").isNull(), 44.44).otherwise(col("spd_mean")))
                    .withColumn("high_spd_count", when(col("high_spd_count").isNull(), 0).otherwise(col("high_spd_count")))
                    .withColumn("high_spd_mean", when(col("high_spd_mean").isNull(), 0).otherwise(col("high_spd_mean")))
                    .withColumn("high_spd_max", when(col("high_spd_max").isNull(), 0).otherwise(col("high_spd_max")))
                    )

trip_session_stats = (trip_session_stats
                    .withColumn("odd_ori_max_spd", col("ori_max_spd") > 80)
                    .withColumn("odd_spd_mean", (col("spd_mean") < 10) | (col("spd_mean") > 80))
                    .withColumn("odd_dist", col("dist") < 1)
                    .withColumn("odd_n_sample", col("n_sample_quality") < 0.5)
                    .withColumn("odd_dur_mins", col("dur_mins") > 120)
                    )
trip_session_stats = trip_session_stats.withColumn("odd", col("odd_ori_max_spd") | col("odd_spd_mean") | col("odd_dist") | col("odd_n_sample") | col("odd_dur_mins"))
# saving
trip_session_stats.write.format("orc").mode('overwrite').save(trip_session_stats_file_path)
trip_session_stats = spark.read.option("inferSchema", True).orc(trip_session_stats_file_path)


valid_trips = trip_session_stats.filter(col("odd") == False)
valid_trips = valid_trips.withColumn("trip_id", col("temp_sess"))
# saving
valid_trips.write.format("orc").mode('overwrite').save(valid_sessions_stats_file_path)
valid_trips = spark.read.option("inferSchema", True).orc(valid_sessions_stats_file_path)

trip_data.printSchema()


trip_data = (gps_data
                .filter((col("temp_sess") != 0))
                .join(valid_trips.select("temp_sess", "trip_id"), ["temp_sess"], how="inner")
                .select("trip_id", "driver", "lat", "lng", "ts_utc", "time_del", "dist", "speed", "orient", "next_dist", "action")
            )
trip_data.write.format("orc").mode('overwrite').save(valid_trip_data_file_path)
trip_data = spark.read.option("inferSchema", True).orc(valid_trip_data_file_path)

print("total source data points: {}\ntotal temp trips: {}\nvalid trips: {}\noutput data points: {}".format(total_source_data_count, trip_session_stats.count(), valid_trips.count(), trip_data.count()))
print("time spent: {} min".format(np.round((datetime.now() - program_start).total_seconds() / 60, 1)))









