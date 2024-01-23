#!/usr/bin/env python

import rospy
import ros_numpy
import numpy as np
from numpy import linalg as LA
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import Image
from geometry_msgs.msg import Vector3
import sensor_msgs.point_cloud2 as pc2
from mpl_toolkits.mplot3d import Axes3D

import time
import math
import matplotlib.pyplot as plt
import pyransac3d as pyrsc
from cv_bridge import CvBridge, CvBridgeError
import cv2


def covariance(X, Y, Z):
    cx = np.mean(X)
    cy = np.mean(Y)
    cz = np.mean(Z)

    cov = np.zeros((3,3))
    length = len(X)

    for i in range(length):

        x = X[i]
        y = Y[i]
        z = Z[i]

        cov += np.array([[ (x-cx)**2     , (x-cx)*(y-cy) , (x-cx)*(z-cz)],
                         [ (y-cy)*(x-cx) , (y-cy)**2     , (y-cy)*(z-cz)],
                         [ (z-cz)*(x-cx) , (z-cz)*(y-cy) , (z-cz)**2    ]])

    cov = cov / length
    return cov


def standardize(x):
    mean = np.mean(x)
    std = np.std(x)

    x = (np.array(x) - mean)/std
    return x.tolist()


def callback_pca(data):
    global new_xs, new_ys, new_zs, centroid, dir, dirs, outliers

    t0 = time.time()

    pc = ros_numpy.numpify(data)
    points=np.zeros((pc.shape[0]*pc.shape[1],3))
    points[:,0]=pc['x'].ravel()
    new_xs = (points[:,0])[np.logical_not(np.isnan(points[:,0]))]

    points[:,1]=pc['y'].ravel()
    new_ys = (points[:,1])[np.logical_not(np.isnan(points[:,1]))]

    points[:,2]=pc['z'].ravel()
    new_zs = (points[:,2])[np.logical_not(np.isnan(points[:,2]))]

    x = np.vstack([new_xs, new_ys, new_zs])
    # x = np.vstack([standardize(new_xs), standardize(new_ys), standardize(new_zs)])
    cov = np.cov(x)
    # cov = covariance(new_xs, new_ys, new_zs)

    w, v = LA.eig(cov)

    dir0 = v[np.argmax(w)]
    dir1 = v[np.argmin(w)]
    dir = dir0

    dirs = [v[0], v[1], v[2]]

    centroid = [np.mean(new_xs), np.mean(new_ys), np.mean(new_zs)]


    dir_msg = Vector3()
    dir_msg.x = dir[0]
    dir_msg.y = dir[1]
    dir_msg.z = dir[2]

    print("w: ", w)
    print("v: ", v)
    t1 = time.time()

    points = []
    for i,_ in enumerate(new_xs):
        points.append([new_xs[i],new_ys[i],new_zs[i]])
    outliers=np.array(points)


def callback_ransac(data):
    global new_xs, new_ys, new_zs, centroid, dir, dirs, plane_points_global, \
           outliers, center, corners_global

    t0 = time.time()

    pc = ros_numpy.numpify(data)
    points=np.zeros((pc.shape[0]*pc.shape[1],3))
    points[:,0]=pc['x'].ravel()
    new_xs = (points[:,0])[np.logical_not(np.isnan(points[:,0]))]

    points[:,1]=pc['y'].ravel()
    new_ys = (points[:,1])[np.logical_not(np.isnan(points[:,1]))]

    points[:,2]=pc['z'].ravel()
    new_zs = (points[:,2])[np.logical_not(np.isnan(points[:,2]))]

    points = []
    for i,_ in enumerate(new_xs):
        points.append([new_xs[i],new_ys[i],new_zs[i]])

    points = np.array(points)

    ratio = 1
    last_num_inliers = 0
    first_iter = True
    plane_points=[]
    # outliers=None
    while True:
        plane1 = pyrsc.Plane()
        best_eq, best_inliers = plane1.fit(points, 0.01)

        if first_iter:
            plane_points.append( points[best_inliers,:] )
            centroid = [np.mean(points[best_inliers,0]),np.mean(points[best_inliers,1]),\
                      np.mean(points[best_inliers,2])]
            dir = best_eq[:3]
            dirval = LA.norm(np.array(dir))
            dir=[elem/dirval for elem in dir]
            if dir[2] < 0: dir=[-elem for elem in dir]

            dirs[0] = dir

            points = np.delete(points, best_inliers, 0)
            last_num_inliers = len(best_inliers)
            first_iter = False
            continue

        ratio = len(best_inliers)/float(last_num_inliers)
        if ratio > 0.3:
            plane_points.append( points[best_inliers,:] )
            points = np.delete(points, best_inliers, 0)
            # last_num_inliers = len(best_inliers)
        else:
            break

    first_plane_points = plane_points[0]
    c = np.array(centroid)
    d = np.array(dir)
    dists = LA.norm(first_plane_points - c, axis=1)
    sorted_dists = np.argsort(dists)[::-1]
    corners = []

    for arg in sorted_dists:

        p = first_plane_points[arg,:]

        if np.abs(np.dot(d,p-c)) / LA.norm(p-c) / LA.norm(d) > 0.1:
            continue

        new_corner = True
        for cor in corners:
            # if np.abs(np.dot(cor-c,p-c)) / LA.norm(p-c) / LA.norm(cor-c) > 0.9:
            #     new_corner = False
            if LA.norm(cor-p) < 0.1:
                new_corner = False
        if new_corner:
            corners.append(p)

        if len(corners) == 4:
            break

    frame_points = []
    for i in range(len(corners)):
        for j in range(len(corners)):
            if i==j: continue

            mid = (corners[i] + corners[j])/2
            if LA.norm(mid-c) < 0.1:
                continue

            is_new=True
            for fp in frame_points:
                if LA.norm(mid-fp) < 0.1:
                    is_new=False
            if is_new:
                frame_points.append(mid)

    leftp = np.argmin([p[0] for p in frame_points])
    dirs[2] = (frame_points[leftp]-c)/LA.norm(frame_points[leftp]-c)
    dirs[1] = np.cross(dirs[2],dirs[0])
    corners_global = np.array(frame_points)

    plane_points_global = plane_points
    outliers = points

def to_direction_vector(center, cx, cy, f):

    w = np.array( [ (center[0] - cx) , (center[1] - cy), f] )
    return w/np.linalg.norm(w)

def img_to_points(img, fov):

    f = (0.5 * img.shape[1] * (1.0 / np.tan(fov/2.0) ) )
    idxs = np.array( np.where(np.isfinite(img)) )

    cx = img.shape[1]/2.0
    cy = img.shape[0]/2.0

    points = []
    for i in range(idxs.shape[1]):
        x_pix = idxs[1,i]
        y_pix = idxs[0,i]

        w = to_direction_vector([x_pix, y_pix], cx, cy, f)
        v_distance = img[idxs[0,i],idxs[1,i]]

        factor = v_distance/w[2]

        position = factor*to_direction_vector([x_pix, y_pix], cx, cy, f)

        points.append(position)

    return np.array(points)

def calc_bulk_pos(img, fov):

    cx = img.shape[1]/2.0
    cy = img.shape[0]/2.0
    f = (0.5 * img.shape[1] * (1.0 / np.tan(fov/2.0) ) )

    idxs = np.array( np.where(np.isfinite(img)) )
    x_pix = np.mean(idxs[1,:])
    y_pix = np.mean(idxs[0,:])

    w = to_direction_vector([x_pix, y_pix], cx, cy, f)
    v_distance = np.mean( img[ np.isfinite(img) ] )

    factor = v_distance/w[2]

    position = factor*to_direction_vector([x_pix, y_pix], cx, cy, f)

    return position

def DCM_from_unit_vectors(v1, v2, v3):

    DCMts = np.zeros((3,3))
    DCMts[0,:] = np.array(v1)
    DCMts[1,:] = np.array(v2)
    DCMts[2,:] = np.array(v3)

    return DCMts

def callback_image_ransac(data):
    global new_xs, new_ys, new_zs, centroid, dir, dirs, plane_points_global, \
           outliers, center, corners_global, bridge, fov, cv_image, dcm_pub, \
           pos_pub1, pos_pub2, bulk_pos

    cv_image = bridge.imgmsg_to_cv2(data, "32FC1")
    cv_image_array = np.array(cv_image, dtype = np.dtype('f8'))
    cv_image_norm = cv2.normalize(cv_image_array, cv_image_array, 0, 1, cv2.NORM_MINMAX)
    cv2.imshow("Kinnect image", 1-cv_image_norm)
    cv2.waitKey(1)

    focal_length = (0.5 * cv_image.shape[1] * (1.0 / np.tan(fov/2.0) ) );
    block_size = 0.03

    img_copy=cv_image.copy()
    img_copy[np.isnan(img_copy)] = -1
    dist=np.mean(img_copy[img_copy!=-1])

    if not np.isfinite(dist): return

    dist=np.max([0.2,dist])
    img_block_size = math.ceil(focal_length * block_size /dist)

    des_size = (int(cv_image.shape[1]/img_block_size), int(cv_image.shape[0]/img_block_size))
    cv_image_resized = cv2.resize(cv_image, des_size, interpolation = cv2.INTER_CUBIC)

    bulk_pos = calc_bulk_pos(cv_image_resized, fov)
    if np.isfinite(bulk_pos[0]) and np.isfinite(bulk_pos[1]) and np.isfinite(bulk_pos[2]):
        bposMsg = Vector3()
        bposMsg.x = bulk_pos[0]
        bposMsg.y = bulk_pos[1]
        bposMsg.z = bulk_pos[2]
        pos_pub1.publish(bposMsg)

    points = img_to_points(cv_image_resized, fov)

    ratio = 1
    last_num_inliers = 0
    first_iter = True
    plane_points=[]
    # outliers=None
    while True:
        try:
            plane1 = pyrsc.Plane()
            best_eq, best_inliers = plane1.fit(points, 0.05)
        except Exception as e:
            print("RANSAC failed.")
            return

        if first_iter:
            plane_points.append( points[best_inliers,:] )
            centroid = [np.mean(points[best_inliers,0]),np.mean(points[best_inliers,1]),\
                      np.mean(points[best_inliers,2])]
            dir = best_eq[:3]
            dirval = LA.norm(np.array(dir))
            dir=[elem/dirval for elem in dir]
            if dir[2] < 0: dir=[-elem for elem in dir]

            dirs[0] = dir

            points = np.delete(points, best_inliers, 0)
            last_num_inliers = len(best_inliers)
            first_iter = False
            continue

        ratio = len(best_inliers)/float(last_num_inliers)
        if ratio > 0.3:
            plane_points.append( points[best_inliers,:] )
            points = np.delete(points, best_inliers, 0)
            # last_num_inliers = len(best_inliers)
        else:
            break

    first_plane_points = plane_points[0]
    c = np.array(centroid)
    d = np.array(dir)
    dists = LA.norm(first_plane_points - c, axis=1)
    sorted_dists = np.argsort(dists)[::-1]
    corners = []

    for arg in sorted_dists:

        p = first_plane_points[arg,:]

        if np.abs(np.dot(d,p-c)) / LA.norm(p-c) / LA.norm(d) > 0.1:
            continue

        new_corner = True
        for cor in corners:
            # if np.abs(np.dot(cor-c,p-c)) / LA.norm(p-c) / LA.norm(cor-c) > 0.9:
            #     new_corner = False
            if LA.norm(cor-p) < 0.1:
                new_corner = False
        if new_corner:
            corners.append(p)

        if len(corners) == 4:
            break

    frame_points = []
    for i in range(len(corners)):
        for j in range(len(corners)):
            if i==j: continue

            mid = (corners[i] + corners[j])/2
            if LA.norm(mid-c) < 0.1:
                continue

            is_new=True
            for fp in frame_points:
                if LA.norm(mid-fp) < 0.1:
                    is_new=False
            if is_new:
                frame_points.append(mid)

    leftp = np.argmin([p[0] for p in frame_points])
    dirs[2] = (frame_points[leftp]-c)/LA.norm(frame_points[leftp]-c)
    dirs[1] = np.cross(dirs[2],dirs[0])
    corners_global = np.array(frame_points)

    DCMts = DCM_from_unit_vectors(dirs[0],dirs[1],dirs[2])
    arrMsg = Float32MultiArray()
    arrMsg.data = DCMts.ravel().tolist()
    dcm_pub.publish(arrMsg)

    pposMsg = Vector3()
    pposMsg.x = centroid[0]
    pposMsg.y = centroid[1]
    pposMsg.z = centroid[2]
    pos_pub2.publish(pposMsg)

    plane_points_global = plane_points
    outliers = points


new_xs = []
new_ys = []
new_zs = []
centroid = [0, 0 ,0]
dir = [0, 0, 0]
dirs = [dir, dir, dir]
plane_points_global = []
colors=['green','red','blue','purple','yellow']
outliers = None
center = None
corners_global = None
bulk_pos = [0,0,0]
bridge = CvBridge()
fov=1.047198
cv_image = None

rospy.init_node('pcl_to_array', anonymous=True)
# rospy.Subscriber("/camera/depth/points", PointCloud2, callback_ransac)
rospy.Subscriber("/camera/depth/points", PointCloud2, callback_pca)
# rospy.Subscriber("/camera/depth/image_raw", Image, callback_image_ransac)
dcm_pub = rospy.Publisher('dcm_ts', Float32MultiArray, queue_size=1)
pos_pub1 = rospy.Publisher('target_position/bulk', Vector3, queue_size=1)
pos_pub2 = rospy.Publisher('target_position/precise', Vector3, queue_size=1)

fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(-45, -90)

# rospy.spin()

while not rospy.is_shutdown():

    if outliers is not None:
        plt.cla()
        ax.scatter(outliers[:,0], outliers[:,1], outliers[:,2], c='black', s=3, linewidths=0.0)
        # ax.scatter(corners_global[:,0], corners_global[:,1], corners_global[:,2], c='red', s=1000, marker="*", linewidths=0.0)
        ax.scatter(bulk_pos[0], bulk_pos[1], bulk_pos[2], c='red', s=1000, marker="*", linewidths=0.0)
        for i,pps in enumerate(plane_points_global):
            ax.scatter(pps[:,0], pps[:,1], pps[:,2], c=colors[i], s=3, linewidths=0.0)

        ax.quiver(centroid[0], centroid[1], centroid[2], dirs[0][0], dirs[0][1], dirs[0][2], length=1, linewidth=2, color='red', pivot='tail')
        ax.quiver(centroid[0], centroid[1], centroid[2], dirs[1][0], dirs[1][1], dirs[1][2], length=1, linewidth=2, color='green', pivot='tail')
        ax.quiver(centroid[0], centroid[1], centroid[2], dirs[2][0], dirs[2][1], dirs[2][2], length=1, linewidth=2, color='blue', pivot='tail')
        # ax.quiver(centroid[0], centroid[1], centroid[2], dir[0], dir[1], dir[2], length=1, linewidth=2, color='red')
        plt.draw()

        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        ax.set_xlim(-1,3)
        ax.set_ylim(-2,3)
        ax.set_zlim(7,12)

    plt.pause(1)
