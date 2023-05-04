import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.impute import SimpleImputer


data = pd.read_csv('iceland_discontinuities.dat', sep=' ', skiprows=3, header=None,
                   names=['Quality', 'Station', 'Latitude', 'Longitude', 'Depth'])
data['Depth'] = data['Depth'].astype(float)


# data.drop_duplicates(subset=['Longitude', 'Latitude'], inplace=True)

if data.shape[0] == 0:
    print('No valid data remaining.')
else:
    # Preprocessing depth missing value
    imputer = SimpleImputer(strategy='mean')
    imp_depth = np.array(data['Depth']).reshape(-1, 1)
    imp_depth = imputer.fit_transform(imp_depth)
    data['Depth'] = imp_depth.flatten()

    # Process other NaN values that may exist
    data.dropna(inplace=True)
    # print(data.columns)
    # print(data.isnull().sum())
    # print("---")
    # print(data['Depth'].dtype)
    # print(data.dtypes)
    # Sort the data by depth
    data = data.iloc[np.argsort(data['Depth'].values), :]

    mid_depth = np.median(data['Depth'])

    # Split the data into two subsets, one containing data from the top to the middle tier and the other containing data from the middle tier to the bottom
    df1 = data[data['Depth'] <= mid_depth]
    df2 = data[data['Depth'] > mid_depth]

    # Perform DBSCAN clustering
    eps = 0.05
    min_samples = 15
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(data[['Longitude', 'Latitude']])

    # Draw a scatter plot
    plt.scatter(data['Longitude'].values, data['Latitude'].values, c=clusters, cmap='jet')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Observed interfaces')
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data['Longitude'].values, data['Latitude'].values, data['Depth'].values, c=clusters, cmap='jet')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_zlabel('Depth')
    ax.set_title('Observed interfaces')
    plt.show()

    # Construct a crustal structure model and determine the boundary/transition location between the two and three layers
    depth = data['Depth']
    index = np.argwhere(np.diff(np.sign(depth - np.median(depth))) != 0).flatten() + 1
    boundaries = depth[index[0]], depth[index[-1]]
    boundary_lat = data[data['Depth'] == boundaries[0]]['Latitude'].values[0]
    boundary_lon = data[data['Depth'] == boundaries[0]]['Longitude'].values[0]
    print(
        "The boundary/transition between two and three layers is located at Latitude = {:.2f}, Longitude = {:.2f}, and Depth = {:.2f}. \n".format(
            boundary_lat, boundary_lon, boundaries[0]))
    print("The depth range for the top to middle layer is {:.2f} km to {:.2f} km.".format(min(df1['Depth']),
                                                                                          max(df1['Depth'])))
    print("The depth range for the middle to bottom layer is {:.2f} km to {:.2f} km.".format(min(df2['Depth']),
                                                                                             max(df2['Depth'])))

    # Draw cross sections of North-South and East-West
    fig, axs = plt.subplots(2, 1, figsize=(8, 10))
    axs[0].plot(df1['Latitude'], df1['Depth'], 'b-', label='Top to Middle Layer')
    axs[0].plot(df2['Latitude'], df2['Depth'], 'r-', label='Middle to Bottom Layer')
    axs[0].set_xlabel('Latitude')
    axs[0].set_ylabel('Depth')
    axs[0].legend()
    axs[0].vlines(boundary_lat, 0, depth.iloc[-1], color='black', ls='--', label='Boundary between two & three layers')
    axs[0].text(boundary_lat + 0.05, (boundaries[0] + boundaries[1]) / 2, 'Three layers', fontsize=12)

    axs[1].plot(df1['Longitude'], df1['Depth'], 'b-', label='Top to Middle Layer')
    axs[1].plot(df2['Longitude'], df2['Depth'], 'r-', label='Middle to Bottom Layer')
    axs[1].set_xlabel('Longitude')
    axs[1].set_ylabel('Depth')
    axs[1].legend()
    axs[1].vlines(boundary_lon, 0, depth.iloc[-1], color='black', ls='--', label='Boundary between two & three layers')
    axs[1].text(boundary_lon + 0.05, (boundaries[0] + boundaries[1]) / 2, 'Three layers', fontsize=12)
    plt.savefig("1.png", dpi=300)
    plt.show()
