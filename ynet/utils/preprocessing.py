import numpy as np
import pandas as pd
import os
import cv2
from copy import deepcopy
import math
import matplotlib.pyplot as plt
import pathlib

def load_sdd_raw(path):
	'''
	Loads data from Stanford Drone Dataset. Makes the following preprocessing:
	-filter out unnecessary columns (e.g. generated, label, occluded)
	-filter out non-pedestrian
	-filter out tracks which are lost
	-calculate middle point of bounding box
	-makes new unique, scene-dependent ID (column 'metaId') since original dataset resets id for each scene
	-add scene name to column for visualization
	-output has columns=['trackId', 'frame', 'x', 'y', 'sceneId', 'metaId']

	before data needs to be in the following folder structure
	data/SDD/mode               mode can be 'train','val','test'
	|-bookstore_0
		|-annotations.txt
		|-reference.jpg
	|-scene_name
		|-...
	:param path: path to folder, default is 'data/SDD'
	:param mode: dataset split - options['train', 'test', 'val']
	:return: DataFrame containing all trajectories from dataset split
	'''
	data_path = os.path.join(path, "annotations")
	scenes_main = os.listdir(data_path)
	SDD_cols = ['trackId', 'xmin', 'ymin', 'xmax', 'ymax', 'frame', 'lost', 'occluded', 'generated', 'label']
	data = []
	for scene_main in sorted(scenes_main):
		scene_main_path = os.path.join(data_path, scene_main)
		for scene_sub in sorted(os.listdir(scene_main_path)):
			scene_path = os.path.join(scene_main_path, scene_sub)
			annot_path = os.path.join(scene_path, 'annotations.txt')
			scene_df = pd.read_csv(annot_path, header=0, names=SDD_cols, delimiter=' ')
			# Calculate center point of bounding box
			scene_df['x'] = (scene_df['xmax'] + scene_df['xmin']) / 2
			scene_df['y'] = (scene_df['ymax'] + scene_df['ymin']) / 2
			# scene_df = scene_df[scene_df['label'] == 'Pedestrian']  # drop non-pedestrians
			scene_df = scene_df[scene_df['lost'] == 0]  # drop lost samples
			scene_df = scene_df.drop(columns=['xmin', 'xmax', 'ymin', 'ymax', 'occluded', 'generated', 'lost'])
			scene_df['sceneId'] = f"{scene_main}_{scene_sub.split('video')[1]}"
			# new unique id by combining scene_id and track_id
			scene_df['rec&trackId'] = [recId + '_' + str(trackId).zfill(4) for recId, trackId in
									zip(scene_df.sceneId, scene_df.trackId)]
			data.append(scene_df)
	data = pd.concat(data, ignore_index=True)
	rec_trackId2metaId = {}
	for i, j in enumerate(data['rec&trackId'].unique()):
		rec_trackId2metaId[j] = i
	data['metaId'] = [rec_trackId2metaId[i] for i in data['rec&trackId']]
	data = data.drop(columns=['rec&trackId'])
	return data


def load_SDD(path='data/SDD/', mode='train'):
	'''
	Loads data from Stanford Drone Dataset. Makes the following preprocessing:
	-filter out unnecessary columns (e.g. generated, label, occluded)
	-filter out non-pedestrian
	-filter out tracks which are lost
	-calculate middle point of bounding box
	-makes new unique, scene-dependent ID (column 'metaId') since original dataset resets id for each scene
	-add scene name to column for visualization
	-output has columns=['trackId', 'frame', 'x', 'y', 'sceneId', 'metaId']

	before data needs to be in the following folder structure
	data/SDD/mode               mode can be 'train','val','test'
	|-bookstore_0
		|-annotations.txt
		|-reference.jpg
	|-scene_name
		|-...
	:param path: path to folder, default is 'data/SDD'
	:param mode: dataset split - options['train', 'test', 'val']
	:return: DataFrame containing all trajectories from dataset split
	'''
	assert mode in ['train', 'val', 'test']

	path = os.path.join(path, mode)
	scenes = os.listdir(path)
	SDD_cols = ['trackId', 'xmin', 'ymin', 'xmax', 'ymax', 'frame', 'lost', 'occluded', 'generated', 'label']
	data = []
	print('loading ' + mode + ' data')
	for scene in scenes:
		scene_path = os.path.join(path, scene, 'annotations.txt')
		scene_df = pd.read_csv(scene_path, header=0, names=SDD_cols, delimiter=' ')
		# Calculate center point of bounding box
		scene_df['x'] = (scene_df['xmax'] + scene_df['xmin']) / 2
		scene_df['y'] = (scene_df['ymax'] + scene_df['ymin']) / 2
		scene_df = scene_df[scene_df['label'] == 'Pedestrian']  # drop non-pedestrians
		scene_df = scene_df[scene_df['lost'] == 0]  # drop lost samples
		scene_df = scene_df.drop(columns=['xmin', 'xmax', 'ymin', 'ymax', 'occluded', 'generated', 'label', 'lost'])
		scene_df['sceneId'] = scene
		# new unique id by combining scene_id and track_id
		scene_df['rec&trackId'] = [recId + '_' + str(trackId).zfill(4) for recId, trackId in
								   zip(scene_df.sceneId, scene_df.trackId)]
		data.append(scene_df)
	data = pd.concat(data, ignore_index=True)
	rec_trackId2metaId = {}
	for i, j in enumerate(data['rec&trackId'].unique()):
		rec_trackId2metaId[j] = i
	data['metaId'] = [rec_trackId2metaId[i] for i in data['rec&trackId']]
	data = data.drop(columns=['rec&trackId'])
	return data

def mask_step(x, step):
	"""
	Create a mask to only contain the step-th element starting from the first element. Used to downsample
	"""
	mask = np.zeros_like(x)
	mask[::step] = 1
	return mask.astype(bool)


def downsample(df, step):
	"""
	Downsample data by the given step. Example, SDD is recorded in 30 fps, with step=30, the fps of the resulting
	df will become 1 fps. With step=12 the result will be 2.5 fps. It will do so individually for each unique
	pedestrian (metaId)
	:param df: pandas DataFrame - necessary to have column 'metaId'
	:param step: int - step size, similar to slicing-step param as in array[start:end:step]
	:return: pd.df - downsampled
	"""
	mask = df.groupby(['metaId'])['metaId'].transform(mask_step, step=step)
	return df[mask]


def filter_short_trajectories(df, threshold):
	"""
	Filter trajectories that are shorter in timesteps than the threshold
	:param df: pandas df with columns=['x', 'y', 'frame', 'trackId', 'sceneId', 'metaId']
	:param threshold: int - number of timesteps as threshold, only trajectories over threshold are kept
	:return: pd.df with trajectory length over threshold
	"""
	len_per_id = df.groupby(by='metaId', as_index=False).count()  # sequence-length for each unique pedestrian
	idx_over_thres = len_per_id[len_per_id['frame'] >= threshold]  # rows which are above threshold
	idx_over_thres = idx_over_thres['metaId'].unique()  # only get metaIdx with sequence-length longer than threshold
	df = df[df['metaId'].isin(idx_over_thres)]  # filter df to only contain long trajectories
	return df


def groupby_sliding_window(x, window_size, stride):
	x_len = len(x)
	n_chunk = (x_len - window_size) // stride + 1
	idx = []
	metaId = []
	for i in range(n_chunk):
		idx += list(range(i * stride, i * stride + window_size))
		metaId += ['{}_{}'.format(x.metaId.unique()[0], i)] * window_size
	# temp = x.iloc()[(i * stride):(i * stride + window_size)]
	# temp['new_metaId'] = '{}_{}'.format(x.metaId.unique()[0], i)
	# df = df.append(temp, ignore_index=True)
	df = x.iloc()[idx]
	df['newMetaId'] = metaId
	return df


def sliding_window(df, window_size, stride):
	"""
	Assumes downsampled df, chunks trajectories into chunks of length window_size. When stride < window_size then
	chunked trajectories are overlapping
	:param df: df
	:param window_size: sequence-length of one trajectory, mostly obs_len + pred_len
	:param stride: timesteps to move from one trajectory to the next one
	:return: df with chunked trajectories
	"""
	gb = df.groupby(['metaId'], as_index=False)
	df = gb.apply(groupby_sliding_window, window_size=window_size, stride=stride)
	df['metaId'] = pd.factorize(df['newMetaId'], sort=False)[0]
	df = df.drop(columns='newMetaId')
	df = df.reset_index(drop=True)
	return df


def split_at_fragment_lambda(x, frag_idx, gb_frag):
	""" Used only for split_fragmented() """
	metaId = x.metaId.iloc()[0]
	counter = 0
	if metaId in frag_idx:
		split_idx = gb_frag.groups[metaId]
		for split_id in split_idx:
			x.loc[split_id:, 'newMetaId'] = '{}_{}'.format(metaId, counter)
			counter += 1
	return x


def split_fragmented(df):
	"""
	Split trajectories when fragmented (defined as frame_{t+1} - frame_{t} > 1)
	Formally, this is done by changing the metaId at the fragmented frame and below
	:param df: DataFrame containing trajectories
	:return: df: DataFrame containing trajectories without fragments
	"""

	gb = df.groupby('metaId', as_index=False)
	# calculate frame_{t+1} - frame_{t} and fill NaN which occurs for the first frame of each track
	df['frame_diff'] = gb['frame'].diff().fillna(value=1.0).to_numpy()
	fragmented = df[df['frame_diff'] != 1.0]  # df containing all the first frames of fragmentation
	gb_frag = fragmented.groupby('metaId')  # helper for gb.apply
	frag_idx = fragmented.metaId.unique()  # helper for gb.apply
	df['newMetaId'] = df['metaId']  # temporary new metaId

	df = gb.apply(split_at_fragment_lambda, frag_idx, gb_frag)
	df['metaId'] = pd.factorize(df['newMetaId'], sort=False)[0]
	df = df.drop(columns='newMetaId')
	return df

def load_raw_dataset(step, window_size, stride, path=None, mode='train', pickle_path=None,
							  train_labels=[], test_labels=[], test_per=1.0, max_train_agents=10000,
							  train_set_ratio=1.0, test_on_train=False, num_train_agents=None, num_test_agents=None,
							  random_train_test=True):
	"""
	Helper function to aggregate loading and preprocessing in one function. Preprocessing contains:
	- Split fragmented trajectories
	- Downsample fps
	- Filter short trajectories below threshold=window_size
	- Sliding window with window_size and stride
	:param step (int): downsample factor, step=30 means 1fps and step=12 means 2.5fps on SDD
	:param window_size (int): Timesteps for one window
	:param stride (int): How many timesteps to stride in windowing. If stride=window_size then there is no overlap
	:param path (str): Path to SDD directory (not subdirectory, which is contained in mode)
	:param mode (str): Which dataset split, options=['train', 'val', 'test']
	:param pickle_path (str): Alternative to path+mode, if there already is a pickled version of the raw SDD as df
	:return pd.df: DataFrame containing the preprocessed data
	"""
	if pickle_path is not None:
		df = pd.read_pickle(pickle_path)
	else:
		df = load_sdd_raw(path=path)
	df = split_fragmented(df)  # split track if frame is not continuous
	df = downsample(df, step=step)
	df = filter_short_trajectories(df, threshold=window_size)
	df = sliding_window(df, window_size=window_size, stride=stride)
	df_train = filter_labels(df, train_labels)
	if test_labels == train_labels:
		df_train, df_test = split_df(df_train, ratio=train_set_ratio, test_on_train=test_on_train,
									 num_train_agents=num_train_agents, num_test_agents=num_test_agents,
									 random_train_test=random_train_test)
	else:
		df_test = filter_labels(df, test_labels)
	df_train, df_test = reduce_least_occuring_label(df_train, df_test, test_per, max_train_agents, num_test_agents=num_test_agents)
	# df_train = df_train.drop(columns=['label'])
	# df_test= df_test.drop(columns=['label'])
	return df_train, df_test

def load_and_window_SDD(step, window_size, stride, path=None, mode='train', pickle_path=None):
	"""
	Helper function to aggregate loading and preprocessing in one function. Preprocessing contains:
	- Split fragmented trajectories
	- Downsample fps
	- Filter short trajectories below threshold=window_size
	- Sliding window with window_size and stride
	:param step (int): downsample factor, step=30 means 1fps and step=12 means 2.5fps on SDD
	:param window_size (int): Timesteps for one window
	:param stride (int): How many timesteps to stride in windowing. If stride=window_size then there is no overlap
	:param path (str): Path to SDD directory (not subdirectory, which is contained in mode)
	:param mode (str): Which dataset split, options=['train', 'val', 'test']
	:param pickle_path (str): Alternative to path+mode, if there already is a pickled version of the raw SDD as df
	:return pd.df: DataFrame containing the preprocessed data
	"""
	if pickle_path is not None:
		df = pd.read_pickle(pickle_path)
	else:
		df = load_SDD(path=path, mode=mode)
	df = split_fragmented(df)  # split track if frame is not continuous
	df = downsample(df, step=step)
	df = filter_short_trajectories(df, threshold=window_size)
	df = sliding_window(df, window_size=window_size, stride=stride)

	return df

def filter_labels(df, labels):
	if labels == []:
		return None
	else:
		return df[np.array([df['label'].values == label for label in labels]).any(axis=0)]

def reduce_least_occuring_label(df_train, df_test, test_per, max_train_agents, num_test_agents=None):
	if df_train is not None:
		labels = np.unique(df_train["label"].values)
		# Filter based on the least occuring label across all scenes
		min_num_train = min([len(np.unique(df_train[df_train["label"] == label]["metaId"].values))
					for label in labels] + [0 if max_train_agents is None else max_train_agents])
		meta_ids_keep = []
		for label in labels:
			meta_ids = np.unique(df_train[df_train["label"] == label]["metaId"].values)
			mask = np.zeros_like(meta_ids).astype(bool)
			mask[:min_num_train] = True
			np.random.shuffle(mask)
			meta_ids_keep.append(meta_ids[mask])
		meta_ids_keep = np.array(meta_ids_keep).reshape(-1)
		df_train = df_train[np.array([df_train["metaId"] == meta_id for meta_id in meta_ids_keep]).any(axis=0)]
	else:
		min_num_train = 0
	# Filter test set based on given percentage
	if df_test is not None:
		meta_ids = np.unique(df_test["metaId"].values)
		labels = np.unique(df_test["label"]) 
		num_agents_tot = [len(df_test[df_test["label"] == label]) for label in labels]
		num_steps_agent = round(sum(num_agents_tot)/len(meta_ids))
		min_num_label = min(num_agents_tot)//num_steps_agent if num_test_agents is None else num_test_agents
		meta_ids_final = np.array([np.unique(df_test[df_test["label"] == label]["metaId"].values)[:min_num_label]
									for label in labels]).reshape(-1)
		df_test = df_test[np.array([df_test["metaId"] == meta_id for meta_id in meta_ids_final]).any(axis=0)]
		print(f"{min_num_train} agents for each training class, {min_num_label} agents for test class")
	return df_train, df_test

def split_df(df, ratio=None, test_on_train=False, num_train_agents=None, num_test_agents=None, random=True, random_train_test=True):
	random_train_test = True
	meta_ids = np.unique(df["metaId"].values)
	if ratio is not None:
		num_test = int(len(meta_ids)*(1-ratio))
	elif num_test_agents is not None:
		num_test = num_test_agents
	else:
		raise ValueError

	if test_on_train and num_train_agents is None and num_test_agents is not None:
		mask = np.zeros_like(meta_ids).astype(bool)
		mask[:num_test] = True
		split_mask = np.array([df["metaId"] == meta_id for meta_id in meta_ids[mask]]).any(axis=0)
		return df, df[split_mask]
	else:
		mask = np.ones_like(meta_ids).astype(bool)
		mask[:num_test] = False
		if random_train_test:
			np.random.shuffle(mask)
		split_mask = np.array([df["metaId"] == meta_id for meta_id in meta_ids[mask]]).any(axis=0)

	if test_on_train and num_train_agents is None and num_test_agents is None:
		return df, df[split_mask == False]
	elif test_on_train and num_train_agents is not None and num_test_agents is not None:
		mask_train = np.zeros_like(meta_ids).astype(bool)
		train_idx_all = np.where(mask)[0]
		if random_train_test:
			train_idx = np.random.choice(train_idx_all, num_train_agents)
		else:
			train_idx = train_idx_all[:num_train_agents]
		mask_train[train_idx] = True
		split_mask_train = np.array([df["metaId"] == meta_id for meta_id in meta_ids[mask_train]]).any(axis=0)
		return df[split_mask_train], df[split_mask == False]
	else:
		return df[split_mask], df[split_mask == False]


def rot(df, image, k=1):
	'''
	Rotates image and coordinates counter-clockwise by k * 90° within image origin
	:param df: Pandas DataFrame with at least columns 'x' and 'y'
	:param image: PIL Image
	:param k: Number of times to rotate by 90°
	:return: Rotated Dataframe and image
	'''
	xy = df.copy()
	if image.ndim == 3:
		y0, x0, channels = image.shape
	else:
		y0, x0= image.shape

	xy.loc()[:, 'x'] = xy['x'] - x0 / 2
	xy.loc()[:, 'y'] = xy['y'] - y0 / 2
	c, s = np.cos(-k * np.pi / 2), np.sin(-k * np.pi / 2)
	R = np.array([[c, s], [-s, c]])
	xy.loc()[:, ['x', 'y']] = np.dot(xy[['x', 'y']], R)
	for i in range(k):
		image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

	if image.ndim == 3:
		y0, x0, channels = image.shape
	else:
		y0, x0= image.shape

	xy.loc()[:, 'x'] = xy['x'] + x0 / 2
	xy.loc()[:, 'y'] = xy['y'] + y0 / 2
	return xy, image


def fliplr(df, image):
	'''
	Flip image and coordinates horizontally
	:param df: Pandas DataFrame with at least columns 'x' and 'y'
	:param image: PIL Image
	:return: Flipped Dataframe and image
	'''
	xy = df.copy()
	if image.ndim == 3:
		y0, x0, channels = image.shape
	else:
		y0, x0= image.shape

	xy.loc()[:, 'x'] = xy['x'] - x0 / 2
	xy.loc()[:, 'y'] = xy['y'] - y0 / 2
	R = np.array([[-1, 0], [0, 1]])
	xy.loc()[:, ['x', 'y']] = np.dot(xy[['x', 'y']], R)
	image = cv2.flip(image, 1)

	if image.ndim == 3:
		y0, x0, channels = image.shape
	else:
		y0, x0= image.shape

	xy.loc()[:, 'x'] = xy['x'] + x0 / 2
	xy.loc()[:, 'y'] = xy['y'] + y0 / 2
	return xy, image


def augment_data(data, image_path='data/SDD/train', images={}, image_file='reference.jpg', seg_mask=False, use_raw_data=False):
	'''
	Perform data augmentation
	:param data: Pandas df, needs x,y,metaId,sceneId columns
	:param image_path: example - 'data/SDD/val'
	:param images: dict with key being sceneId, value being PIL image
	:param image_file: str, image file name
	:param seg_mask: whether it's a segmentation mask or an image file
	:return:
	'''
	ks = [1, 2, 3]
	for scene in data.sceneId.unique():
		scene_name, scene_idx = scene.split("_")
		if use_raw_data:
			im_path = os.path.join(image_path, scene_name, f"video{scene_idx}", image_file)
		else:
			im_path = os.path.join(image_path, scene, image_file)
		if seg_mask:
			im = cv2.imread(im_path, 0)
		else:
			im = cv2.imread(im_path)
		images[scene] = im
	data_ = data.copy()  # data without rotation, used so rotated data can be appended to original df
	k2rot = {1: '_rot90', 2: '_rot180', 3: '_rot270'}
	for k in ks:
		metaId_max = data['metaId'].max()
		for scene in data_.sceneId.unique():
			if use_raw_data:
				im_path = os.path.join(image_path, scene_name, f"video{scene_idx}", image_file)
			else:
				im_path = os.path.join(image_path, scene, image_file)
			if seg_mask:
				im = cv2.imread(im_path, 0)
			else:
				im = cv2.imread(im_path)

			data_rot, im = rot(data_[data_.sceneId == scene], im, k)
			# image
			rot_angle = k2rot[k]
			images[scene + rot_angle] = im

			data_rot['sceneId'] = scene + rot_angle
			data_rot['metaId'] = data_rot['metaId'] + metaId_max + 1
			data = data.append(data_rot)

	metaId_max = data['metaId'].max()
	for scene in data.sceneId.unique():
		im = images[scene]
		data_flip, im_flip = fliplr(data[data.sceneId == scene], im)
		data_flip['sceneId'] = data_flip['sceneId'] + '_fliplr'
		data_flip['metaId'] = data_flip['metaId'] + metaId_max + 1
		data = data.append(data_flip)
		images[scene + '_fliplr'] = im_flip

	return data, images


def augment_eth_ucy_social(train_batches, train_scenes, train_masks, train_images):
	""" Augment ETH/UCY data that is preprocessed with social masks """
	# Rotate by 90°, 180°, 270°
	train_batches_aug = train_batches.copy()
	train_scenes_aug = train_scenes.copy()
	train_masks_aug = train_masks.copy()
	for scene in np.unique(train_scenes):
		image = train_images[scene].copy()
		for rot_times in range(1, 4):
			scene_trajectories = deepcopy(train_batches)
			scene_trajectories = scene_trajectories[train_scenes == scene]

			rot_angle = 90 * rot_times

			# Get middle point and calculate rotation matrix
			if image.ndim == 3:
				H, W, C = image.shape
			else:
				H, W = image.shape
			c, s = np.cos(-rot_times * np.pi / 2), np.sin(-rot_times * np.pi / 2)
			R = np.array([[c, s], [-s, c]])
			middle = np.array([W, H]) / 2

			# rotate image
			image_rot = image.copy()
			for _ in range(rot_times):
				image_rot = cv2.rotate(image_rot, cv2.ROTATE_90_COUNTERCLOCKWISE)
			if image_rot.ndim == 3:
				H, W, C = image_rot.shape
			else:
				H, W = image_rot.shape
			# get new rotated middle point
			middle_rot = np.array([W, H]) / 2
			# perform transformation on trajectories
			for traj in scene_trajectories:
				# substract middle
				traj[:, :, 2:4] -= middle
				traj[:, :, 2:4] = np.dot(traj[:, :, 2:4], R)
				traj[:, :, 2:4] += middle_rot

			train_images[f'{scene}_{rot_angle}'] = image_rot
			train_batches_aug = np.append(train_batches_aug, scene_trajectories, axis=0)
			train_scenes_aug = np.append(train_scenes_aug,
										 np.array([f'{scene}_{rot_angle}'] * scene_trajectories.shape[0]), axis=0)
			train_masks_aug = np.append(train_masks_aug, train_masks[train_scenes == scene], axis=0)

	# Flip
	train_batches = deepcopy(train_batches_aug)
	train_scenes = deepcopy(train_scenes_aug)
	train_masks = deepcopy(train_masks_aug)
	for scene in np.unique(train_scenes):
		image = train_images[scene].copy()
		scene_trajectories = deepcopy(train_batches)
		scene_trajectories = scene_trajectories[train_scenes == scene]

		# Get middle point and calculate rotation matrix
		if image.ndim == 3:
			H, W, C = image.shape
		else:
			H, W = image.shape
		R = np.array([[-1, 0], [0, 1]])
		middle = np.array([W, H]) / 2

		# rotate image
		image_rot = image.copy()
		image_rot = cv2.flip(image_rot, 1)
		if image_rot.ndim == 3:
			H, W, C = image_rot.shape
		else:
			H, W = image_rot.shape
		# get new rotated middle point
		middle_rot = np.array([W, H]) / 2
		# perform transformation on trajectories
		for traj in scene_trajectories:
			# substract middle
			traj[:, :, 2:4] -= middle
			traj[:, :, 2:4] = np.dot(traj[:, :, 2:4], R)
			traj[:, :, 2:4] += middle_rot

		train_images[f'{scene}_flip'] = image_rot
		train_batches_aug = np.append(train_batches_aug, scene_trajectories, axis=0)
		train_scenes_aug = np.append(train_scenes_aug, np.array([f'{scene}_flip'] * scene_trajectories.shape[0]),
									 axis=0)
		train_masks_aug = np.append(train_masks_aug, train_masks[train_scenes == scene], axis=0)
	return train_batches_aug, train_scenes_aug, train_masks_aug


def resize_and_pad_image(images, size, pad=2019):
	""" Resize image to desired size and pad image to make it square shaped and ratio of images still is the same, as
	images all have different sizes.
	"""
	for key, im in images.items():
		H, W, C = im.shape
		im = cv2.copyMakeBorder(im, 0, pad - H, 0, pad - W, cv2.BORDER_CONSTANT)
		im = cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)
		images[key] = im


def create_images_dict(data, image_path, image_file='reference.jpg', use_raw_data=False):
	images = {}
	for scene in data.sceneId.unique():
		if image_file == 'oracle.png':
			im = cv2.imread(os.path.join(image_path, scene, image_file), 0)
		else:
			if use_raw_data:
				scene_name, scene_idx = scene.split("_")
				im_path = os.path.join(image_path, scene_name, f"video{scene_idx}", image_file)
			else:
				im_path = os.path.join(image_path, scene, image_file)
			im = cv2.imread(im_path)
		images[scene] = im
	return images


def load_images(scenes, image_path, image_file='reference.jpg'):
	images = {}
	if type(scenes) is list:
		scenes = set(scenes)
	for scene in scenes:
		if image_file == 'oracle.png':
			im = cv2.imread(os.path.join(image_path, scene, image_file), 0)
		else:
			im = cv2.imread(os.path.join(image_path, scene, image_file))
		images[scene] = im
	return images


def read_trajnet(mode='train'):
	root = 'data/SDD_trajnet/'
	path = os.path.join(root, mode)

	fp = os.listdir(path)
	df_list = []
	for file in fp:
		name = file.split('.txt')[0]

		df = pd.read_csv(os.path.join(path, file), sep=' ', names=['frame', 'trackId', 'x', 'y'])
		df['sceneId'] = name
		df_list.append(df)

	df = pd.concat(df_list, ignore_index=True)
	df['metaId'] = [recId + '_' + str(trackId).zfill(4) for recId, trackId in zip(df.sceneId, df.trackId)]
	df['metaId'] = pd.factorize(df['metaId'], sort=False)[0]
	return df


def load_inD(path='data/inD/', scenes=[1], recordings=None):
	'''
	Loads data from inD Dataset. Makes the following preprocessing:
	-filter out unnecessary columns
	-filter out non-pedestrian
	-makes new unique ID (column 'metaId') since original dataset resets id for each scene
	-add scene name to column for visualization
	-output has columns=['trackId', 'frame', 'x', 'y', 'sceneId', 'metaId']

	data needs to be in the following folder structure
	data/inD/*.csv

	:param path: str - path to folder, default is 'data/inD'
	:param scenes: list of integers - scenes to load
	:param recordings: list of strings - alternative to scenes, load specified recordings instead, overwrites scenes
	:return: DataFrame containing all trajectories from split
	'''

	scene2rec = {1: ['00', '01', '02', '03', '04', '05', '06'],
				 2: ['07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17'],
				 3: ['18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29'],
				 4: ['30', '31', '32']}

	rec_to_load = []
	for scene in scenes:
		rec_to_load.extend(scene2rec[scene])
	if recordings is not None:
		rec_to_load = recordings
	data = []
	for rec in rec_to_load:
		# load csv
		track = pd.read_csv(os.path.join(path, '{}_tracks.csv'.format(rec)))
		track = track.drop(columns=['trackLifetime', 'heading', 'width', 'length', 'xVelocity', 'yVelocity',
									'xAcceleration', 'yAcceleration', 'lonVelocity', 'latVelocity',
									'lonAcceleration', 'latAcceleration'])
		track_meta = pd.read_csv(os.path.join(path, '{}_tracksMeta.csv'.format(rec)))

		# Filter non-pedestrians
		pedestrians = track_meta[track_meta['class'] == 'pedestrian']
		track = track[track['trackId'].isin(pedestrians['trackId'])]

		track['rec&trackId'] = [str(recId) + '_' + str(trackId).zfill(6) for recId, trackId in
								zip(track.recordingId, track.trackId)]
		track['sceneId'] = rec
		track['yCenter'] = -track['yCenter']
		data.append(track)

	data = pd.concat(data, ignore_index=True)

	rec_trackId2metaId = {}
	for i, j in enumerate(data['rec&trackId'].unique()):
		rec_trackId2metaId[j] = i
	data['metaId'] = [rec_trackId2metaId[i] for i in data['rec&trackId']]
	data = data.drop(columns=['rec&trackId', 'recordingId'])
	data = data.rename(columns={'xCenter': 'x', 'yCenter': 'y'})

	cols_order = ['trackId', 'frame', 'x', 'y', 'sceneId', 'metaId']
	data = data.reindex(columns=cols_order)
	return data


def load_and_window_inD(step, window_size, stride, scenes=[1,2,3,4], pickle=False):
	"""
	Helper function to aggregate loading and preprocessing in one function. Preprocessing contains:
	- Split fragmented trajectories
	- Downsample fps
	- Filter short trajectories below threshold=window_size
	- Sliding window with window_size and stride
	:param step (int): downsample factor, step=30 means 1fps and step=12 means 2.5fps on SDD
	:param window_size (int): Timesteps for one window
	:param stride (int): How many timesteps to stride in windowing. If stride=window_size then there is no overlap
	:param scenes (list of int): Which scenes to load, inD has 4 scenes
	:param pickle (Bool): If True, load pickle instead of csv
	:return pd.df: DataFrame containing the preprocessed data
	"""
	df = load_inD(path='data/inD/', scenes=scenes, recordings=None)
	df = downsample(df, step=step)
	df = filter_short_trajectories(df, threshold=window_size)
	df = sliding_window(df, window_size=window_size, stride=stride)

	return df

def compute_vel_labels(df):
	vel_labels = {}
	meta_ids = np.unique(df["metaId"].values)
	for meta_id in meta_ids:
		vel_mean, label = compute_vel_meta_id(df, meta_id)
		if label not in vel_labels:
			vel_labels[label] = []
		vel_labels[label] += [vel_mean]
	return vel_labels

def compute_vel_meta_id(df, meta_id):
	df_meta = df[df["metaId"] == meta_id]
	x = df_meta["x"].values
	y = df_meta["y"].values
	unique_labels = np.unique(df_meta["label"].values)
	assert len(unique_labels) == 1
	label = unique_labels[0]
	frame_steps = []
	for frame_idx, frame in enumerate(df_meta["frame"].values):
		if frame_idx != len(df_meta["frame"].values) - 1:
			frame_steps.append(df_meta["frame"].values[frame_idx + 1] - frame)
	unique_frame_step = np.unique(frame_steps)
	assert len(unique_frame_step) == 1
	frame_step = unique_frame_step[0]
	vel = []
	for i in range(len(x)):
		if i != len(x) - 1:
			vel_i = math.sqrt(((x[i+1] - x[i])/frame_step)**2 + ((y[i+1] - y[i])/frame_step)**2)
			vel.append(vel_i)
	vel_mean = np.mean(vel)
	return vel_mean, label

def create_vel_dataset(df, vel_ranges, labels, out_dir):
	pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
	vel_meta_ids = {vel_range: {"metaId": [], "sceneId": [], "label": []} for vel_range in vel_ranges}
	meta_ids = np.unique(df["metaId"].values)
	for meta_id in meta_ids:
		vel_mean, label = compute_vel_meta_id(df, meta_id)
		if label not in labels:
			continue
		for vel_range in vel_meta_ids.keys():
			vel_min, vel_max = vel_range
			if vel_mean >= vel_min and vel_mean <= vel_max:
				vel_meta_ids[vel_range]["metaId"].append(meta_id)
				unique_scene_ids = np.unique(df[df["metaId"] == meta_id]["sceneId"].values)
				assert len(unique_scene_ids) == 1
				scene_id = unique_scene_ids[0]
				vel_meta_ids[vel_range]["sceneId"].append(scene_id)
				vel_meta_ids[vel_range]["label"].append(label)
	
	num_metas = min([len(vel_range_meta_ids["metaId"]) for vel_range_meta_ids in vel_meta_ids.values()])
	for vel_range, vel_range_meta_ids in vel_meta_ids.items():
		scene_ids, scene_counts = np.unique(vel_range_meta_ids["sceneId"], return_counts=True)
		sorted_unique_scene_counts = np.unique(np.sort(scene_counts))
		total_count = 0
		prev_count = 0
		mask = np.zeros_like(scene_counts).astype(bool)
		for scene_count in sorted_unique_scene_counts:
			total_count += (scene_counts >= scene_count).sum() * (scene_count - prev_count)
			if total_count >= num_metas:
				break
			mask[scene_counts == scene_count] = True
			prev_count = scene_count
		total_counts = np.zeros_like(scene_counts)
		total_counts[mask] = scene_counts[mask]
		total_counts[mask==False] = prev_count
		less = True
		while less:
			for i in np.where(mask==False)[0]:
				total_counts[i] += min(1, num_metas - total_counts.sum())
				if num_metas == total_counts.sum():
					less = False
					break
		vel_range_meta_ids["sceneId"] = np.array(vel_range_meta_ids["sceneId"])
		vel_range_meta_ids["metaId"] = np.array(vel_range_meta_ids["metaId"])
		vel_range_meta_ids["label"] = np.array(vel_range_meta_ids["label"])
		meta_id_mask = np.zeros_like(vel_range_meta_ids["metaId"]).astype(bool)
		for scene_idx, scene_id in enumerate(scene_ids):
			scene_count = total_counts[scene_idx]
			scene_mask = vel_range_meta_ids["sceneId"] == scene_id
			scene_labels = vel_range_meta_ids["label"][scene_mask]
			unique_scene_labels, scene_labels_count = np.unique(scene_labels, return_counts=True)
			scene_labels_chosen = []
			while len(scene_labels_chosen) < scene_count:
				for label_idx, (unique_scene_label, scene_label_count) in enumerate(zip(unique_scene_labels, scene_labels_count)):
					if scene_label_count != 0:
						scene_labels_chosen.append(unique_scene_label)
						scene_labels_count[label_idx] -= 1
						if len(scene_labels_chosen) == scene_count:
							break
			labels_chosen, labels_chosen_count = np.unique(scene_labels_chosen, return_counts=True)
			for label, label_count in zip(labels_chosen, labels_chosen_count):
				meta_id_idx = np.where(np.logical_and(vel_range_meta_ids["label"] == label, vel_range_meta_ids["sceneId"] == scene_id))[0][:label_count]
				meta_id_mask[meta_id_idx] = True
		df_vel = df[np.array([df["metaId"] == meta_id for meta_id in vel_range_meta_ids["metaId"][meta_id_mask]]).any(axis=0)]
		vel_range_name = f"{vel_range[0]}_{vel_range[1]}"
		df_vel["vel_range"] = np.array(vel_range_name).repeat(len(df_vel))
		out_path = os.path.join(out_dir, f"{vel_range_name}.pkl")
		df_vel.to_pickle(out_path)

def create_vel_histograms(df, out_dir):
	vel_labels = compute_vel_labels(df)
	vel_all = []
	# Visualize data
	for label, vel_label in vel_labels.items():
		if label not in ["Pedestrian", "Biker"]:
			continue
		plot_histogram(vel_label, label, out_dir)
		vel_all += vel_label
	plot_histogram(vel_all, "All", out_dir)

def plot_histogram(vel, label, out_dir):
	fig = plt.figure()
	mean = np.round(np.mean(vel), 2)
	std = np.round(np.std(vel), 2)
	min_val = np.round(np.min(vel), 2)
	max_val = np.round(np.max(vel), 2)
	num_zeros = np.round((np.array(vel) == 0).sum()/len(vel), 2)
	vel_label = np.sort(vel)[int(len(vel)*0.00):int(len(vel)*0.99)]
	vel_label = vel_label[vel_label != 0]
	plt.hist(vel_label, bins=4)
	plt.title(f"{label}, Mean: {mean}, Std: {std}, Min: {min_val}, Max: {max_val}, Zeros: {num_zeros}")
	pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
	plt.savefig(os.path.join(out_dir, label))
	plt.close(fig)

if __name__ == "__main__":
	# TRAIN_DATA_PATH = FOLDERNAME + 'dataset_custom/2022_01_29_19_41_47_train.pkl'
	# VAL_DATA_PATH = FOLDERNAME + 'dataset_custom/2022_01_29_19_41_47_val.pkl'
	# TRAIN_DATA_PATH = FOLDERNAME + 'ynet_additional_files/data/SDD/train_trajnet.pkl'
	# VAL_DATA_PATH = FOLDERNAME + 'dataset_custom/2022_01_29_19_41_47_val.pkl'

	# Visualize dataset
	FOLDERNAME = "/fastdata/vilab07/sdd/"
	DATA_PATH = FOLDERNAME + 'dataset_biker/gap/2.25_2.75.pkl'
	OUT_DIR = "./visu/vel/complete_test"
	df = pd.read_pickle(DATA_PATH)
	print("Length", len(df))
	create_vel_histograms(df, OUT_DIR)

	# Create dataset
	# FOLDERNAME = "/fastdata/vilab07/sdd/"
	# data_path = FOLDERNAME + 'dataset_custom/complete.pkl'
	# out_dir = FOLDERNAME + 'dataset_biker/gap'
	# vel_ranges = [(0.25, 0.75), (1.25, 1.75), (2.25, 2.75), (3.25, 3.75)]
	# # vel_ranges = [(0.5, 1.5), (1.5, 2.5), (2.5, 3.5), (3.5, 4.5)]
	# df = pd.read_pickle(data_path)
	# # labels = ["Pedestrian", "Biker"]
	# labels = ["Biker"]
	# create_vel_dataset(df, vel_ranges, labels, out_dir)