def write_belief_map(keypoints, layer_res):

  num_features = keypoints.shape(0)
  belief_map = np.zeroes((layer_res, layer_res, num_features), float)

  for i in range (num_features):
    belief_map[ : :i] = keypoint_gaussian(belief_map[ : :i], keypoints[i])

  return belief_map


def keypoint_gaussian(image, kp, var = 1) :
  y_bound, x_bound = image.shape[:2]

  # Calculate range of gaussian 
  top_left = [int(kp[0] - 3 * var), 
              int(kp[1] - 3 * var)]
  bottom_right = [int(kp[0] + 3 * var + 1), 
                  int(kp[1] + 3 * var + 1)]
  
  # Check if entire gaussian distribution is out of image range
  if top_left[0] > x_bound or top_left[1] > y_bound \
    or bottom_right[0] < 0 or bottom_right[0] < 0:
    return image

  # Compute 2 Dimensional Gaussian (un-normalized)
  range = 6 * var + 1
  x = np.arange(range, dtype=float)
  x_c = x - (range//2)
  y_c = x_c[:, np.newaxis]
  g = np.exp(- (x_c ** 2 + y_c ** 2) / (2 * var ** 2))

  # gaussian range in image
  g_x = max(0, -top_left[0]), min(bottom_right[0], x_bound) - top_left[0]
  g_y = max(0, -top_left[1]), min(bottom_right[1], y_bound) - top_left[1]

  # Image range
  i_x = max(0, top_left[0]), min(bottom_right[0], x_bound)
  i_y = max(0, top_left[1]), min(bottom_right[1], y_bound)

  # Copy gaussian to image
  image [i_y[0]:i_y[1], i_x[0]:i_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
  return image