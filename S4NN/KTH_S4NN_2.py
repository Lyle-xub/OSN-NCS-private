import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

data = np.load("/home/ubuntu/zf/KTH/hdf5_files/b-hc-hw_dev.npy", allow_pickle=True)
# data = np.load('/home/ubuntu/zf/KTH/hdf5_files/j-r-w_dev.npy', allow_pickle=True)

labels = data[:, 4].astype(int)
video_order = data[:, 5]
video_name = data[:, 6]


file_path = "/home/ubuntu/zf/KTH/hdf5_files/hc-hw_output.npy"
file_path2 = "/home/ubuntu/zf/KTH/hdf5_files/b-hc_output.npy"
file_path3 = "/home/ubuntu/zf/KTH/hdf5_files/b-hw_output.npy"

# file_path = '/home/ubuntu/zf/KTH/hdf5_files/r-w_output.npy'
# file_path2 = '/home/ubuntu/zf/KTH/hdf5_files/j-r_output.npy'
# file_path3 = '/home/ubuntu/zf/KTH/hdf5_files/j-w_output.npy'


video_array = np.load(file_path)
video_array2 = np.load(file_path2)
video_array3 = np.load(file_path3)


result_array = np.zeros((len(video_array), 3), dtype=int)

for i, value in enumerate(video_array):
    if value == 0:
        result_array[i, 1] += 1
    elif value == 1:
        result_array[i, 2] += 1

for i, value in enumerate(video_array2):
    if value == 0:
        result_array[i, 0] += 1
    elif value == 1:
        result_array[i, 1] += 1

for i, value in enumerate(video_array3):
    if value == 0:
        result_array[i, 0] += 1
    elif value == 1:
        result_array[i, 2] += 1

max_indices = np.argmax(result_array, axis=1)
transformed_array = np.zeros((len(max_indices), 3), dtype=int)
for i, value in enumerate(max_indices):
    transformed_array[i, value] = 1

accuracy_frame = accuracy_score(labels, max_indices)
conf_mat = confusion_matrix(labels, max_indices)

conf_mat_percentage = conf_mat / conf_mat.sum(axis=1)[:, np.newaxis] * 100

blue_cmap = LinearSegmentedColormap.from_list("blue_cmap", ["white", "blue"])
plt.figure(figsize=(8, 6))
ax = sns.heatmap(
    conf_mat_percentage,
    annot=True,
    fmt=".2f",
    cmap=blue_cmap,
    cbar=False,
    annot_kws={"size": 20},
    square=True,
)


for _, spine in ax.spines.items():
    spine.set_visible(True)
    spine.set_linewidth(2)
    spine.set_color("black")


plt.title(f"Confusion Matrix (frame_accuracy: {accuracy_frame*100:.2f}%)", fontsize=16)
plt.xlabel("Predicted Labels", fontsize=14)
plt.ylabel("True Labels", fontsize=14)
plt.tight_layout()
plt.show()


paired_data = zip(video_name, transformed_array)
accumulated_vectors = {}


for name, vector in paired_data:
    if name in accumulated_vectors:
        accumulated_vectors[name] += vector
    else:
        accumulated_vectors[name] = np.array(vector, dtype=int)


video_output = []
for name, vector in accumulated_vectors.items():
    max_index = np.argmax(vector)
    video_output.append(max_index)

print("Max indices for all videos:", video_output)
frame_accuracy = f"Confusion Matrix (frame_accuracy: {accuracy_frame})"
print(frame_accuracy)


unique_labels = {}
for name, label in zip(video_name, labels):
    if name not in unique_labels:
        unique_labels[name] = label

ordered_labels = np.array([unique_labels[name] for name in unique_labels])
print("Ordered unique labels for each unique video name:", ordered_labels)


# Enabled when calculating video accuracy
conf_mat = confusion_matrix(ordered_labels, video_output)
accuracy = accuracy_score(ordered_labels, video_output)
blue_cmap = LinearSegmentedColormap.from_list("blue_cmap", ["white", "blue"])
plt.figure(figsize=(8, 6))
ax = sns.heatmap(
    conf_mat,
    annot=True,
    fmt=".2f",
    cmap=blue_cmap,
    cbar=False,
    annot_kws={"size": 20},
    square=True,
)
plt.title(f"Confusion Matrix (Accuracy: {accuracy*100:.2f}%)", fontsize=16)
plt.xlabel("Predicted Labels", fontsize=14)
plt.ylabel("True Labels", fontsize=14)
plt.tight_layout()
plt.show()


conf_mat_percentage = (
    conf_mat.astype("float") / conf_mat.sum(axis=1)[:, np.newaxis] * 100
)
video_accuracy = accuracy_score(ordered_labels, video_output)

blue_cmap = LinearSegmentedColormap.from_list("blue_cmap", ["white", "blue"])
plt.figure(figsize=(8, 6))
ax = sns.heatmap(
    conf_mat_percentage,
    annot=True,
    fmt=".2f",
    cmap=blue_cmap,
    cbar=False,
    annot_kws={"size": 20},
    square=True,
)

for _, spine in ax.spines.items():
    spine.set_visible(True)
    spine.set_linewidth(2)
    spine.set_color("black")


plt.title(f"Confusion Matrix (video_accuracy: {video_accuracy*100:.2f}%)", fontsize=16)
plt.xlabel("Predicted Labels", fontsize=14)
plt.ylabel("True Labels", fontsize=14)
plt.tight_layout()
plt.show()

Video_accuracy = f"Confusion Matrix (Video_accuracy: {video_accuracy})"
print(Video_accuracy)
