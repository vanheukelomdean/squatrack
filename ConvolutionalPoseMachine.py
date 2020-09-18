STAGE_1_LAYERS = 11
STAGE_T_INPUT_LAYERS = 7
STAGE_T_CONCAT_LAYERS = 5

class ConvPoseMachine (keras.models.Sequential):
  def __init__(self, num_stages, num_parts):
      super(ConvPoseMachine, self).__init__()
      assert num_stages > 1, "There must be at least an initial and subsequent stage"
      self.num_stages = num_stages
      self.num_parts = num_parts
      self.stage = 0
      self.loss = 0
      self.total_loss = 0
      self.build_model()
      
  def build_model(self):
    filters = self.num_parts + 1
    layers  = []
    # Initial stage
    # Normalize
    layers.append(experimental.preprocessing.Rescaling(1./255))

    # Triple 9x9 conv, 5x5 conv
    for i in range (3):
      layers.append(Conv2D(filters,
                            name="conv-1-" + str(i + 1),
                            kernel_size=[9, 9],
                            activation='relu'))
      layers.append(MaxPooling2D(name="pool-1-" + str(i + 1),
                                 pool_size=(2, 2)))

    layers.append(Conv2D(filters,
                          name="conv-1-4", 
                          kernel_size=[5, 5],
                          activation='relu'))
    
    # Finishing 11x11 conv, double 1x1 conv
    layers.append(Conv2D(filters,
                          name="conv-1-5", 
                          kernel_size=[9, 9],
                          activation='relu'))
    layers.append(Conv2D(filters,
                          name="conv-1-6",
                          kernel_size=[1, 1],
                          activation='relu'))
    layers.append(Conv2D(filters,
                          name="conv-1-7",
                          kernel_size=[1, 1],
                          activation='relu'))
    layers.append(StageLoss(parent_model=self,
                            num_outputs=filters))

    # Subsequent Stages

    for i in range (1, self.num_stages):
      # Triple 9x9 conv, 5x5 conv
      for i in range (3):
        layers.append(Conv2D(filters,
                              kernel_size=[9, 9],
                              activation='relu'))
        layers.append(MaxPooling2D(pool_size=(2, 2)))

      layers.append(Conv2D(filters,
                            kernel_size=[5, 5],
                            activation='relu'))
      
      # Concat belief map and previous stage feature map
      for i in range (3):
        layers.append(Conv2D(filters,
                              kernel_size=[11, 11],
                              activation='relu'))
      layers.append(Conv2D(filters,
                            kernel_size=[1, 1],
                            activation='relu'))
      layers.append(Conv2D(filters,
                            kernel_size=[1, 1],
                            activation='relu'))
      layers.append(StageLoss(parent_model=self,
                              num_outputs=filters))

    self.nn_layers = layers

  def call(self, img, x_n = None):
    global STAGE_1_LAYERS, STAGE_T_INPUT_LAYERS, STAGE_T_CONCAT_LAYERS
    if self.stage == 0:
      for l in range (0, STAGE_1_LAYERS):
        print(l)
        img = self.nn_layers[l](img)
      return img

    else: 
      for l in range (self.stage, self.stage + STAGE_T_INPUT_LAYERS):
        img = self.layers[l](img)
    
      maps = tf.concat(img, x_n)

      for l in range (self.stage, self.stage + STAGE_T_CONCAT_LAYERS):
        maps = self.nn_layers[l](maps)
      return maps

  def setLoss(self, stage_loss):
    assert stage_loss > 0, "Stage loss less than zero"
    self.loss = value
    self.total_loss += self.loss