import numpy as np
from scipy.stats import rankdata
import matplotlib.pyplot as plt
import cv2


def scale_data(input_tensor):
    min_val = np.min(input_tensor)
    max_val = np.max(input_tensor)
    scaled_tensor = (input_tensor - min_val) / (max_val - min_val)
    return scaled_tensor


track = np.load('track_char1.npy')
idx = 200
location = track[0,idx,:]    # idx 坐标
image = cv2.imread('mask_char1_00001.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print(image.shape)# (540,960,3) ,
color = image[round(location[1]),round(location[0]),:]
point_h = []
point_w = []
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        if image[i,j,0] == color[0] and image[i,j,1] == color[1] and image[i,j,2] == color[2]:
            #print(i,j)
            point_h.append(i)
            point_w.append(j)
#print(point_h,point_w)
object = np.concatenate((np.array(point_h).reshape(-1,1), np.array(point_w).reshape(-1,1)),axis=1)
print(object.shape)

point_h = []
point_w = []
fit_num = 0
for i in range(track.shape[1]):
    found = np.any(np.all( np.array([round(track[0,i,1]),round(track[0,i,0])]).reshape(1,2) == object, axis=1))

    if found:
        fit_num+=1
        point_h.append(track[0,i,1])
        point_w.append(track[0,i,0])
fit_point = np.concatenate((np.array(point_h).reshape(-1,1), np.array(point_w).reshape(-1,1)),axis=1)
print(fit_point.shape)
print(fit_num)


mi_matrix = np.load(f'mi_char1_{idx}.npy')
print((mi_matrix))

location = track[0]
'''
point = location[400]
print(location.shape, location[:10, :])
image = cv2.imread('/home/kangsong/video_seg_2/picture.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.scatter(location[:, 0], location[:, 1], c='red', cmap='jet', s=10)
plt.scatter(point[0], point[1], c='black', cmap='jet', s=100)
plt.show()
'''
print(mi_matrix.shape)
num_all = []
num_fit = []
for i in np.arange(1, 0, -0.01):
    relation = (mi_matrix)
    # max = np.max(relation)
    # relation[i] = max + 0.1
    relation = scale_data(rankdata(relation))
    #print("For data point", i, "MI is", relation)
    # relation = relation.reshape(10,10)
    num_ob = 0

    num = 0

    for j in range(relation.shape[0]):
        if relation[j] >= i:
            num += 1
            found_ob = np.any(np.all( np.array([round(location[j,1]),round(location[j,0])]).reshape(1,2) == object, axis=1))
            found_fit = np.any(
                np.all(np.array([location[j,1],location[j,0]]).reshape(1,2) == fit_point, axis=1))
            if found_ob:
                num_ob+=1
    print(num_ob,  num)
    num_all.append((num_ob)/fit_num)
    num_fit.append(num_ob/num)

np.save(f'recall4_mi{idx}.npy',num_all)
np.save(f'precision4_mi{idx}.npy',num_fit)
plt.style.use("ggplot")
fig = plt.figure(figsize=(14, 14))

ax1 = fig.add_subplot(111)

ax1.plot(num_all, num_fit, color="red", lw=2, ls="-", label="precision-recall", markersize=10)

ax1.set_xlabel("recall", fontweight="bold", fontsize=20)
ax1.set_ylabel(" precision ", fontweight="bold", fontsize=20)

ax1.legend(fontsize=20, loc="lower left")
#plt.show()
plt.savefig(f'100-mi{idx}.png')

