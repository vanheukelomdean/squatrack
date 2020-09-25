MPII_FIELDS = ['NAME','r ankle_X','r ankle_Y', 'r knee_X','r knee_Y', 'r hip_X',
               'r hip_Y', 'l hip_X','l hip_Y', 'l knee_X','l knee_Y', 'l ankle_X',
               'l ankle_Y','pelvis_X','pelvis_Y','thorax_X','thorax_Y','upper neck_X',
               'upper neck_Y', 'head top_X','head top_Y', 'r wrist_X','r wrist_Y',
               'r elbow_X','r elbow_Y', 'r shoulder_X','r shoulder_Y','l shoulder_X',
               'l shoulder_Y','l elbow_X','l elbow_Y', 'l wrist_X','l wrist_Y','Scale',
               'Activity','Category']

DATASET_SIZE = 24984 / 100

class MpiiDataset():
  def __init__(self, image_dir, annotation_path, train=True, shuffle=True, csv=False, filter = []):
    self.network_input_dim = 368
    self.train = train
    self.shuffle = shuffle
    self.image_dir = image_dir
    self.bboxes = {}

    print("Loading annotations...")

    pb = ProgressBar(total=100, decimals=0, length=50, fill='X', zfill='-')

    release = sio.loadmat(annotation_path, struct_as_record=False)['RELEASE']

    print("Transforming annotations...")

    obj = release[0,0]

    annolist = obj.annolist
    train_flags = obj.img_train
    act = obj.act

    self.labels = pd.DataFrame(columns=MPII_FIELDS)

    # for each annotated image record
    for i in range(0,annolist.shape[1]):

      # Only save training or test images
      if not train_flags[0,i] == self.train:
        continue
        
      temp = []
      obj_list = annolist[0,i]
      obj_act = act[i,0]
      
      rect =obj_list.__dict__['annorect']
      img_d = obj_list.__dict__['image']

      if rect.shape[0] == 0:
        continue
          
      obj_rect = rect[0,0]
      obj_img = img_d[0,0]

      if 'annopoints' not in obj_rect._fieldnames:
        continue

      
      # Write image name to record
      name = obj_img.__dict__['name'][0]
      annopoints = obj_rect.__dict__['annopoints']
      
      if annopoints.shape[0]==0:
        continue
        
      if not filter == [] and not name in filter:
        continue

      points = annopoints[0,0].__dict__['point']

      temp.append(name)
    
      # Set default keypoint coordinate value -1
      for n in range(0,32):
        temp.append(-1)

      keypoints = []
      # Write keypoints to record
      for px in range(0,points.shape[1]):
        point = points[0,px]
        id = point.__dict__['id']
        x = point.__dict__['x']
        y = point.__dict__['y']
        array_index = 2 * id[0][0] + 1
        temp[array_index] = x[0][0]
        temp[array_index+1] = y[0][0]
        keypoints.append((x[0][0], y[0][0]))
      
      # Store bboxes in seperate map from dataframe
      self.bboxes[str(name)] = BoundingBox(keypoints)

      # Write ratio of box size to 200px height
      scale = obj_rect.__dict__['scale'][0][0]
      temp.append(scale)

      # Write activity/category, take the first index if passed list
      activity = act[i,0]
      activity_name = activity.act_name
      category_name = activity.cat_name

      if activity_name.shape[0]==0:
          temp.append(activity_name)
      else:
          temp.append(activity_name[0])
      if category_name.shape[0]==0:
          temp.append(category_name)
      else:
          temp.append(category_name[0])

      self.labels = pd.concat([self.labels, pd.DataFrame([temp],columns=MPII_FIELDS)])

      pb.print_progress_bar(int(i / DATASET_SIZE)) 
      
    print("\n" + ("Training" if self.train else "Testing") + " annotations dataframe (size " + str(self.labels.shape) + ") loaded")

    if (csv):
      file_name = "train" if self.train else "test" + '_mpii.csv'
      data.to_csv(file_name)
      print("Dataset written to " + file_name)


  def preprocess_image(self, image_name):

    image_path = join(image_dir, image_name)
    image = cv2.imread(image_path)

    # Cache image keypoints and human's bounding box
    label = self.labels[self.labels['NAME'].str.contains(image_name)]
    bbox = self.bboxes[image_name]

    top_left, bottom_right = bbox.tl_br()

    # Scale image to have human roughly 200 px in height
    targetHeight = 200.0
    scalingFactor = targetHeight / bbox.height
    image = cv2.resize(image, (0, 0), fx=scalingFactor, fy=scalingFactor)
    bbox.rescale((scalingFactor, scalingFactor))

    top_left, bottom_right = bbox.tl_br()

    bbox.expand(self.network_input_dim)

    half_cross = np.full((1, 2), self.network_input_dim / 2).astype(int)[0]
    full_cross = np.add(half_cross, half_cross)
    half_cross_tuple = (self.network_input_dim // 2,  self.network_input_dim // 2)

    # Pad image with black
    pad_image = np.pad(image, (half_cross_tuple, half_cross_tuple, (0, 0)), mode='constant')

    # Add margin to bounding box top left for cropping start and image diagonal for cropping end
    start = np.add(bbox.top_left.astype(int), half_cross)
    end =  np.add(start, full_cross)

    # Crop image to network input dimensions with human centered
    crop_image = pad_image[start[1]:end[1], start[0]:end[0]]

    # Perform similar transformations on laabeled annotations
    labelX = (np.array(label.iloc[:, 1:32:2])* scalingFactor + half_cross[0] - start[0]).astype(np.int32)[0]
    labelY = (np.array(label.iloc[:, 2:33:2])* scalingFactor + half_cross[0] - start[1]).astype(np.int32)[0]
    indices = range(0, len(labelX))
    
    transformed_labels = list(map(lambda x: (labelX[x], labelY[x]), indices))
    #transformed_labels = np.hstack([labelY, labelX])

    return crop_image, transformed_labels

  def get_data(self, validation_split=0.2):
    images = [f for f in os.listdir(self.image_dir)if os.path.isfile(os.path.join(self.image_dir, f))]

    # shuffle all labels
    if self.shuffle:
        self.labels = shuffle(self.labels)

    #label_images = self.labels['NAME'].to_list()
    # compute label-image set intersection
    dataset = list (set(self.labels['NAME'].to_list()) & set(images))


    train_len = int(len(dataset) * (1 - validation_split))
    for image_name in dataset[:train_len]:
      yield self.preprocess_image(image_name)