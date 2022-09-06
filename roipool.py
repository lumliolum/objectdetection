import torch
import torchvision

import numpy as np

# feature_map = np.array([
#     [0.70, 0.41, 0.38, 1.23, 0.24, 0.81, 1.19],
#     [0.14, 0.45, 0.31, 0.73, 3.22, 0.64, 0.96],
#     [0.11, 0.41, 0.79, 0.69, 0.44, 0.62, 1.23],
#     [1.47, 0.25, 0.09, 0.32, 2.98, 0.99, 0.35],
#     [0.48, 0.87, 0.77, 0.26, 0.11, 0.05, 0.33],
#     [0.14, 0.45, 0.31, 0.73, 2.22, 0.64, 0.96],
#     [0.11, 0.41, 0.11, 0.69, 0.44, 0.62, 1.23]
# ])

# feature_map = np.array([
#     [0, 1, 2, 3, 4],
#     [5, 6, 7, 8, 9],
#     [10, 11, 12, 13, 14],
#     [15, 16, 17, 18, 19],
#     [20, 21, 22, 23, 24],
# ])

feature_map = np.array([
    [0.70, 0.41, 0.38, 1.23, 0.24],
    [0.14, 0.45, 0.31, 0.73, 3.22],
    [0.11, 0.41, 0.79, 0.69, 0.44],
    [1.47, 0.25, 0.09, 0.32, 2.98],
    [0.48, 0.87, 0.77, 0.26, 0.11],
])

feature_map = torch.tensor(feature_map, requires_grad=True, dtype=torch.float32)

# (batch, channel, h, w) -> (1, 1, 7, 7)
feature_map = feature_map.unsqueeze(0).unsqueeze(0)
feature_map.retain_grad()

boxes = np.array([
    [0, 0, 0, 4, 4],
])
boxes = torch.tensor(boxes, requires_grad=True, dtype=torch.float32)

print(feature_map.shape, boxes.shape)

# roi pooling layer of 2x2
pool = torchvision.ops.roi_pool(input=feature_map, boxes=boxes, output_size=6)
print(pool)
su = torch.sum(pool)
print(su)
su.backward()
print(feature_map.grad)
