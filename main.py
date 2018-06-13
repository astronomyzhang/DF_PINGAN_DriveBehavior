# -*- coding:utf8 -*-
import pandas as pd
import numpy as np
import datetime
import lightgbm as lgb
from math import radians, cos, sin, asin, sqrt
from sklearn.preprocessing import StandardScaler

path_train = "/data/dm/train.csv"  # 训练文件
path_test = "/data/dm/test.csv"  # 测试文件

path_test_out = "model/result.csv"  # 预测结果输出路径为model/xx.csv,有且只能有一个文件并且是CSV格式。

#=============计算地球球面距离=====================
def haversine1(lon1, lat1, lon2, lat2):  # 经度1，纬度1，经度2，纬度2 （十进制度数）
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # 将十进制度数转化为弧度
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine公式
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # 地球平均半径，单位为公里
    return c * r * 1000

#====================zscore标准化===========================
def std_scale(X):
    scaler = StandardScaler().fit(X)
    return scaler.transform(X)


def process():
    """
    处理过程，在示例中，使用随机方法生成结果，并将结果文件存储到预测结果路径下。
    :return:
    """
    train = pd.read_csv(path_train)
    train.columns = ["TERMINALNO", "TIME", "TRIP_ID", "LONGITUDE", "LATITUDE", "DIRECTION", "HEIGHT", "SPEED", "CALLSTATE", "Y"]

    train_post = []
    for item in train['TERMINALNO'].unique():
        temp = train.loc[train['TERMINALNO']==item,:]
        total_call = temp.shape[0]
        temp.index = range(total_call)
        num_of_trips = temp['TRIP_ID'].nunique()
        num_of_records = total_call
        num_of_state_0 = temp.loc[temp['CALLSTATE']==0].shape[0]
        num_of_state_1 = temp.loc[temp['CALLSTATE']==1].shape[0]
        num_of_state_2 = temp.loc[temp['CALLSTATE']==2].shape[0]
        num_of_state_3 = temp.loc[temp['CALLSTATE']==3].shape[0]
        num_of_state_4 = temp.loc[temp['CALLSTATE']==4].shape[0]

        mean_speed = temp['SPEED'].mean()
        var_speed = temp['SPEED'].var()
        mean_height = temp['HEIGHT'].mean()

        temp['hour'] = temp['TIME'].apply(lambda x:datetime.datetime.fromtimestamp(x).hour)
        hour_state = np.zeros(24)
        for i in range(24):
            hour_state[i] = temp[temp['hour']==i].shape[0] / float(total_call)

        target = temp.loc[0, 'Y']

    #=============方向DIRECTION的特征提取====================
        trip_chg = []
        for x in temp['TRIP_ID'].unique():
            direc_data = temp.loc[temp['TRIP_ID']==x, "DIRECTION"]
            gps_data = temp.loc[temp['TRIP_ID']==x, ["LONGITUDE", "LATITUDE"]]
            direc_data.index = range(direc_data.shape[0])
            gps_data.index = range(gps_data.shape[0])
            trip_records = direc_data.shape[0] #number of record in one trip
            direc_chg_total = 0 #trip in total
            dist = 0
            if trip_records > 1:
                for i in range(1,trip_records):
                    direc_chg_total += abs(direc_data[i] - direc_data[i-1])
                    lon1 = gps_data.loc[i, "LONGITUDE"]
                    lat1 = gps_data.loc[i, "LATITUDE"]
                    lon2 = gps_data.loc[i-1, "LONGITUDE"]
                    lat2 = gps_data.loc[i-1, "LATITUDE"]
                    dist += abs(haversine1(lon1, lat1, lon2, lat2))
            direc_chg_avg = float(direc_chg_total) / trip_records
            trip_chg.append([x, direc_chg_avg, dist])
        trip_chg = pd.DataFrame(trip_chg)
        trip_chg.columns = ['TRIP_ID', 'DIERECTION_CHG', 'DIST']
        var_direc = trip_chg['DIERECTION_CHG'].var()
        mean_direc = trip_chg['DIERECTION_CHG'].mean()
        median_dist = trip_chg['DIST'].median()
        var_dist = trip_chg['DIST'].var()
        #=======================================================

        feature = [item, num_of_records, num_of_trips, num_of_state_0, num_of_state_1, num_of_state_2, num_of_state_3, num_of_state_4, \
                   hour_state[0],hour_state[1],hour_state[2],hour_state[3],hour_state[4],hour_state[5],hour_state[6],hour_state[7],\
                   hour_state[8], hour_state[9],hour_state[10],hour_state[11],hour_state[12],hour_state[13],hour_state[14],hour_state[15], \
                   hour_state[16],hour_state[17], hour_state[18],hour_state[19],hour_state[20],hour_state[21],hour_state[22],hour_state[23],\
                   mean_speed, var_speed, mean_height, var_direc, mean_direc, median_dist, var_dist,target]
        train_post.append(feature)

    train_post = pd.DataFrame(train_post)
    feature_name = ['TERMINALNO', 'num_of_records', 'num_of_trips', 'num_of_state_0', 'num_of_state_1', 'num_of_state_2', \
                    'num_of_state_3', 'num_of_state_4','h0','h1', 'h2','h3','h4','h5','h6','h7','h8','h9','h10','h11','h12', \
                    'h13','h14','h15','h16','h17','h18','h19','h20','h21','h22','h23', 'mean_speed', 'var_speed', 'mean_height', \
                    'var_direc', 'mean_direc', 'median_dist', 'var_dist', 'target']
    train_post.columns = feature_name

#========================测试集特征提取==============================
    test = pd.read_csv(path_test)
    test_post = []
    for item in test['TERMINALNO'].unique():
        temp = test.loc[test['TERMINALNO']==item,:]
        total_call = temp.shape[0]
        temp.index = range(total_call)
        num_of_trips = temp['TRIP_ID'].nunique()
        num_of_records = total_call
        num_of_state_0 = temp.loc[temp['CALLSTATE']==0].shape[0]
        num_of_state_1 = temp.loc[temp['CALLSTATE']==1].shape[0]
        num_of_state_2 = temp.loc[temp['CALLSTATE']==2].shape[0]
        num_of_state_3 = temp.loc[temp['CALLSTATE']==3].shape[0]
        num_of_state_4 = temp.loc[temp['CALLSTATE']==4].shape[0]

        mean_speed = temp['SPEED'].mean()
        var_speed = temp['SPEED'].var()
        mean_height = temp['HEIGHT'].mean()

        temp['hour'] = temp['TIME'].apply(lambda x:datetime.datetime.fromtimestamp(x).hour)
        hour_state = np.zeros(24)
        for i in range(24):
            hour_state[i] = temp[temp['hour']==i].shape[0] / float(total_call)

    #=============方向DIRECTION的特征提取====================
        trip_chg = []
        for x in temp['TRIP_ID'].unique():
            direc_data = temp.loc[temp['TRIP_ID']==x, "DIRECTION"]
            gps_data = temp.loc[temp['TRIP_ID']==x, ["LONGITUDE", "LATITUDE"]]
            direc_data.index = range(direc_data.shape[0])
            gps_data.index = range(gps_data.shape[0])
            trip_records = direc_data.shape[0] #number of record in one trip
            direc_chg_total = 0 #trip in total
            dist = 0
            if trip_records > 1:
                for i in range(1,trip_records):
                    direc_chg_total += abs(direc_data[i] - direc_data[i-1])
                    lon1 = gps_data.loc[i, "LONGITUDE"]
                    lat1 = gps_data.loc[i, "LATITUDE"]
                    lon2 = gps_data.loc[i-1, "LONGITUDE"]
                    lat2 = gps_data.loc[i-1, "LATITUDE"]
                    dist += abs(haversine1(lon1, lat1, lon2, lat2))
            direc_chg_avg = float(direc_chg_total) / trip_records
            trip_chg.append([x, direc_chg_avg, dist])
        trip_chg = pd.DataFrame(trip_chg)
        trip_chg.columns = ['TRIP_ID', 'DIERECTION_CHG', 'DIST']
        var_direc = trip_chg['DIERECTION_CHG'].var()
        mean_direc = trip_chg['DIERECTION_CHG'].mean()
        median_dist = trip_chg['DIST'].median()
        var_dist = trip_chg['DIST'].var()
        #=======================================================

        feature = [item, num_of_records, num_of_trips, num_of_state_0, num_of_state_1, num_of_state_2, num_of_state_3, num_of_state_4, \
                   hour_state[0],hour_state[1],hour_state[2],hour_state[3],hour_state[4],hour_state[5],hour_state[6],hour_state[7],\
                   hour_state[8], hour_state[9],hour_state[10],hour_state[11],hour_state[12],hour_state[13],hour_state[14],hour_state[15], \
                   hour_state[16],hour_state[17], hour_state[18],hour_state[19],hour_state[20],hour_state[21],hour_state[22],hour_state[23],\
                   mean_speed, var_speed, mean_height, var_direc, mean_direc, median_dist, var_dist]
        test_post.append(feature)

    test_post = pd.DataFrame(test_post)
    feature_name = ['TERMINALNO', 'num_of_records', 'num_of_trips', 'num_of_state_0', 'num_of_state_1', 'num_of_state_2', \
                    'num_of_state_3', 'num_of_state_4','h0','h1', 'h2','h3','h4','h5','h6','h7','h8','h9','h10','h11','h12', \
                    'h13','h14','h15','h16','h17','h18','h19','h20','h21','h22','h23', 'mean_speed', 'var_speed', 'mean_height', \
                   'var_direc', 'mean_direc', 'median_dist', 'var_dist']
    test_post.columns = feature_name


#=======================================================================================


    feature_use = ['TERMINALNO', 'num_of_records', 'num_of_trips', 'num_of_state_0', 'num_of_state_1', 'num_of_state_2', \
                	'num_of_state_3', 'num_of_state_4','h0','h1', 'h2','h3','h4','h5','h6','h7','h8','h9','h10','h11','h12', \
                	'h13','h14','h15','h16','h17','h18','h19','h20','h21','h22','h23', 'mean_speed', 'var_speed', 'mean_height', \
                	'mean_direc', "var_direc",'median_dist', 'var_dist']

#=============================数据zscore标准化====================
    feature_pre = ["var_direc", 'var_dist']
    test_post[feature_pre] = test_post[feature_pre].fillna(0)
    train_post[feature_pre] = std_scale(train_post[feature_pre].fillna(0))
    test_post[feature_pre] = std_scale(test_post[feature_pre])
#=================================================================

    model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                                  learning_rate=0.01, n_estimators=720,
                                  max_bin = 55, bagging_fraction = 0.8,
                                  bagging_freq = 5, feature_fraction = 0.2319,
                                  feature_fraction_seed=9, bagging_seed=9,
                                  min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

    model_lgb.fit(train_post[feature_use], train_post['target'])
    prediction = model_lgb.predict(test_post[feature_use])

    result = pd.DataFrame({
    'Id':test["TERMINALNO"].unique(),
    'Pred':prediction
    })

#===============Pred 现实意义不能为负数，预测值为负则置为0=========================
    result.loc[result['Pred'] < 0, 'Pred'] = 0.0
#===================================================================================

    result.to_csv("model/result.csv", header=True, index=False)



if __name__ == "__main__":
    print("****************** start **********************")
    # 程序入口
    process()
