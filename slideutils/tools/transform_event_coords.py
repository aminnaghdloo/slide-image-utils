import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# This program intends to find coordinate transform between 10x to 40x images
# using a selection of events and infer the 40x coordinate of the remaining 
# events.
# Inputs:
#		1. cell coordinates in each frame
#		2. frame coordinates from 10x scan
#		3. coordinates of a subset of cells in 40x

# Output:
#		- coordinates of the rest of cells in 40x

# Parameters:
#		- S_x, S_y: 10x pixel size (along x and y)
#		- alpha: transformation angle
#		- X_offset, Y_offset: offset of coordinate system origin
# S_x = 0.590578
# S_y = 0.591877

input_path = "/home/amin/Dropbox/Education/PhD/CSI/Projects/slide-image-utils/tests/map_10x_to_40x/10x_40x_data/0AD7604"
celldata_10x = input_path + "/ocular_celldata_0AD7604.csv"
frame_coords = input_path + "/10x_frame_coords.txt"
celldata_40x = input_path + "/0AD7604_Optimized_Morph_Hit_Locs_80i.csv"

# input_path = "/home/amin/Dropbox/Education/PhD/CSI/Projects/slide-image-utils/tests/map_10x_to_40x/10x_40x_data/0AC1704"
# celldata_10x = input_path + "/ocular_celldata_0AC1704.csv"
# frame_coords = input_path + "/10x_frame_coords.txt"
# celldata_40x = input_path + "/0AC1704_Optimized_Morph_Hit_Locs_80i.csv"

# input_path = "/home/amin/Dropbox/Education/PhD/CSI/Projects/slide-image-utils/tests/map_10x_to_40x/10x_40x_data/0A63004"
# celldata_10x = input_path + "/ocular_celldata_0A63004.csv"
# frame_coords = input_path + "/10x_frame_coords.txt"
# celldata_40x = input_path + "/0A63004_Optimized_Morph_Hit_Locs_Zeiss1.csv"

# input_path = "/home/amin/Dropbox/Education/PhD/CSI/Projects/slide-image-utils/tests/map_10x_to_40x/10x_40x_data/09E3D02"
# celldata_10x = input_path + "/ocular_celldata_09E3D02.csv"
# frame_coords = input_path + "/10x_frame_coords.txt"
# celldata_40x = input_path + "/09E3D02_Optimized_Morph_Hit_Locs_Zeiss1.csv"

# input_path = "/home/amin/Dropbox/Education/PhD/CSI/Projects/slide-image-utils/tests/map_10x_to_40x/10x_40x_data/075469"
# celldata_10x = input_path + "/ocular_celldata_075469.csv"
# frame_coords = input_path + "/10x_frame_coords.txt"
# celldata_40x = input_path + "/075469_Optimized_Morph_Hit_Locs_Zeiss1.csv"

df_10x = pd.read_table(celldata_10x, sep=',').iloc[:,:7]
df_frames = pd.read_table(frame_coords)
df_40x = pd.read_table(celldata_40x, sep=',', header=None)
df_40x.columns = ['cell_id', 'frame_id', 'xx', 'yy', 'reimaging_x', 'reimaging_y']
# df_40x = pd.merge(df_40x, df_10x, on=['frame_id', 'cell_id'], how='inner')
sel_index = df_40x.iloc[:,:3].drop_duplicates().index
df_40x = df_40x.iloc[sel_index,:]
print(df_40x.shape)
print(df_40x)
# temp = pd.merge(df_10x, df_40x, on=['frame_id', 'cell_id'], how='right')
data = pd.merge(df_40x, df_frames, on='frame_id', how='left')
data.index = data.frame_id.astype(str) + '_' + data.cell_id.astype(str)
X = data[['xx', 'yy', 'X_frame', 'Y_frame']]
# X = data[['stagex', 'stagey', 'X_frame', 'Y_frame']]
y = data[['reimaging_x','reimaging_y']] * 1000
print(X)
print(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=20, random_state=21)

# Initialize the multi-output regressor with a linear regression model
multi_target_regressor = MultiOutputRegressor(LinearRegression())

# Train the model
multi_target_regressor.fit(X_train, y_train)

# Make predictions
y_pred = multi_target_regressor.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
dist = np.sqrt(np.sum((y_test - y_pred) ** 2, axis=1))
print("distance data on test:", dist.shape, dist.mean(), dist.max())

print("Mean Squared Error:", mse)
print("R-squared:", r2)


#############
y_train_pred = multi_target_regressor.predict(X_train)
dist = np.sqrt(np.sum((y_train - y_train_pred) ** 2, axis=1))
print("distance data on train:", dist.shape, dist.mean(), dist.max())