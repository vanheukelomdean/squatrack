class BoundingBox():
  def __init__(self, points):
    self.top_left =  np.min(points, axis=0)
    self.bottom_right = np.max(points, axis=0)
    self.update_whc()

  def update_whc(self):
    self.width = np.abs(self.bottom_right[0] - self.top_left[0])
    self.height = np.abs(self.bottom_right[1] - self.top_left[1])
    self.center = np.array(self.top_left[0] + self.width/2, self.top_left[1] + self.height/2)

  def rescale(self, scalingFactor):
    self.top_left = (self.center - scalingFactor * (self.center - self.top_left))
    self.bottom_right = (self.center - scalingFactor * (self.center - self.bottom_right))
    self.update_whc()

  def expand (self, size):
    expansion_vector = np.array([(size - self.width)/2, (size - self.height)/2]) 
    self.top_left -= expansion_vector
    self.bottom_right += expansion_vector
    self.update_whc()

  def tl_br (self):
    return tuple(self.top_left.astype(int)), tuple(self.bottom_right.astype(int))
