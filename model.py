import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from copy import deepcopy

from utils.softargmax import SoftArgmax2D, create_meshgrid
from utils.dataset import augment_data, create_images_dict
from utils.image_utils import create_gaussian_heatmap_template, create_dist_mat, \
	preprocess_image_for_segmentation, pad, resize
from utils.dataloader import SceneDataset, scene_collate
from test import evaluate
from train import train


class StyleModulator(nn.Module):
	def __init__(self, sizes):
		"""
		Additional style modulator for efficient fine-tuning
		"""		
		from ddf import DDFPack
		super(StyleModulator, self).__init__()
		tau = 0.5
		self.modulators = nn.ModuleList(
			[DDFPack(s) for s in sizes + [sizes[-1]]]
		)

	def forward(self, x):
		stylized = []
		for xi, layer in zip(x, self.modulators):
			stylized.append(layer(xi))
		return stylized

class YNetEncoder(nn.Module):
	def __init__(self, in_channels, channels=(64, 128, 256, 512, 512)):
		"""
		Encoder model
		:param in_channels: int, semantic_classes + obs_len
		:param channels: list, hidden layer channels
		"""
		super(YNetEncoder, self).__init__()
		self.stages = nn.ModuleList()

		# First block
		self.stages.append(nn.Sequential(
			nn.Conv2d(in_channels, channels[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			nn.ReLU(inplace=True),
		))

		# Subsequent blocks, each starting with MaxPool
		for i in range(len(channels)-1):
			self.stages.append(nn.Sequential(
				nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
				nn.Conv2d(channels[i], channels[i+1], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
				nn.ReLU(inplace=True),
				nn.Conv2d(channels[i+1], channels[i+1], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
				nn.ReLU(inplace=True)))

		# Last MaxPool layer before passing the features into decoder
		self.stages.append(nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)))

	def forward(self, x):
		# Saves the feature maps Tensor of each layer into a list, as we will later need them again for the decoder
		features = []
		for stage in self.stages:
			x = stage(x)
			features.append(x)
		return features


class YNetDecoder(nn.Module):
	def __init__(self, encoder_channels, decoder_channels, output_len, traj=False):
		"""
		Decoder models
		:param encoder_channels: list, encoder channels, used for skip connections
		:param decoder_channels: list, decoder channels
		:param output_len: int, pred_len
		:param traj: False or int, if False -> Goal and waypoint predictor, if int -> number of waypoints
		"""
		super(YNetDecoder, self).__init__()

		# The trajectory decoder takes in addition the conditioned goal and waypoints as an additional image channel
		if traj:
			encoder_channels = [channel+traj for channel in encoder_channels]
		encoder_channels = encoder_channels[::-1]  # reverse channels to start from head of encoder
		center_channels = encoder_channels[0]

		decoder_channels = decoder_channels

		# The center layer (the layer with the smallest feature map size)
		self.center = nn.Sequential(
			nn.Conv2d(center_channels, center_channels*2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			nn.ReLU(inplace=True),
			nn.Conv2d(center_channels*2, center_channels*2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			nn.ReLU(inplace=True)
		)

		# Determine the upsample channel dimensions
		upsample_channels_in = [center_channels*2] + decoder_channels[:-1]
		upsample_channels_out = [num_channel // 2 for num_channel in upsample_channels_in]

		# Upsampling consists of bilinear upsampling + 3x3 Conv, here the 3x3 Conv is defined
		self.upsample_conv = [
			nn.Conv2d(in_channels_, out_channels_, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
			for in_channels_, out_channels_ in zip(upsample_channels_in, upsample_channels_out)]
		self.upsample_conv = nn.ModuleList(self.upsample_conv)

		# Determine the input and output channel dimensions of each layer in the decoder
		# As we concat the encoded feature and decoded features we have to sum both dims
		in_channels = [enc + dec for enc, dec in zip(encoder_channels, upsample_channels_out)]
		out_channels = decoder_channels

		self.decoder = [nn.Sequential(
			nn.Conv2d(in_channels_, out_channels_, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_channels_, out_channels_, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			nn.ReLU(inplace=True))
			for in_channels_, out_channels_ in zip(in_channels, out_channels)]
		self.decoder = nn.ModuleList(self.decoder)


		# Final 1x1 Conv prediction to get our heatmap logits (before softmax)
		self.predictor = nn.Conv2d(in_channels=decoder_channels[-1], out_channels=output_len, kernel_size=1, stride=1, padding=0)

	def forward(self, features):
		# Takes in the list of feature maps from the encoder. Trajectory predictor in addition the goal and waypoint heatmaps
		features = features[::-1]  # reverse the order of encoded features, as the decoder starts from the smallest image
		center_feature = features[0]
		x = self.center(center_feature)
		for i, (feature, module, upsample_conv) in enumerate(zip(features[1:], self.decoder, self.upsample_conv)):
			x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # bilinear interpolation for upsampling
			x = upsample_conv(x)  # 3x3 conv for upsampling
			x = torch.cat([x, feature], dim=1)  # concat encoder and decoder features
			x = module(x)  # Conv
		x = self.predictor(x)  # last predictor layer
		return x
 

class YNetTorch(nn.Module):
	def __init__(self, obs_len, pred_len, segmentation_model_fp, use_features_only=False, semantic_classes=6,
				 encoder_channels=[], decoder_channels=[], waypoints=1):
		"""
		Complete Y-net Architecture including semantic segmentation backbone, heatmap embedding and ConvPredictor
		:param obs_len: int, observed timesteps
		:param pred_len: int, predicted timesteps
		:param segmentation_model_fp: str, filepath to pretrained segmentation model
		:param use_features_only: bool, if True -> use segmentation features from penultimate layer, if False -> use softmax class predictions
		:param semantic_classes: int, number of semantic classes
		:param encoder_channels: list, encoder channel structure
		:param decoder_channels: list, decoder channel structure
		:param waypoints: int, number of waypoints
		"""
		super(YNetTorch, self).__init__()

		if segmentation_model_fp is not None:
			self.semantic_segmentation = torch.load(segmentation_model_fp)
			if use_features_only:
				self.semantic_segmentation.segmentation_head = nn.Identity()
				semantic_classes = 16  # instead of classes use number of feature_dim
		else:
			self.semantic_segmentation = nn.Identity()


		self.encoder = YNetEncoder(in_channels=semantic_classes + obs_len, channels=encoder_channels)

		self.goal_decoder = YNetDecoder(encoder_channels, decoder_channels, output_len=pred_len)
		self.traj_decoder = YNetDecoder(encoder_channels, decoder_channels, output_len=pred_len, traj=waypoints)

		self.softargmax_ = SoftArgmax2D(normalized_coordinates=False)

		self.encoder_channels = encoder_channels

	def initialize_style(self):
		self.style_modulators = nn.ModuleList([StyleModulator(self.encoder_channels) for _ in range(3)])

	def segmentation(self, image):
		return self.semantic_segmentation(image)

	# Forward pass for goal decoder
	def pred_goal(self, features):
		goal = self.goal_decoder(features)
		return goal

	# Forward pass for trajectory decoder
	def pred_traj(self, features):
		traj = self.traj_decoder(features)
		return traj

	# Forward pass for feature encoder, returns list of feature maps
	def pred_features(self, x):
		features = self.encoder(x)
		return features

	# Used for style encoding
	def stylize_features(self, x, style_class):
		stylized_features = self.style_modulators[style_class](x)
		return stylized_features

	# Softmax for Image data as in dim=NxCxHxW, returns softmax image shape=NxCxHxW
	def softmax(self, x):
		return nn.Softmax(2)(x.view(*x.size()[:2], -1)).view_as(x)

	# Softargmax for Image data as in dim=NxCxHxW, returns 2D coordinates=Nx2
	def softargmax(self, output):
		return self.softargmax_(output)

	def sigmoid(self, output):
		return torch.sigmoid(output)

	def softargmax_on_softmax_map(self, x):
		""" Softargmax: As input a batched image where softmax is already performed (not logits) """
		pos_y, pos_x = create_meshgrid(x, normalized_coordinates=False)
		pos_x = pos_x.reshape(-1)
		pos_y = pos_y.reshape(-1)
		x = x.flatten(2)

		estimated_x = pos_x * x
		estimated_x = torch.sum(estimated_x, dim=-1, keepdim=True)
		estimated_y = pos_y * x
		estimated_y = torch.sum(estimated_y, dim=-1, keepdim=True)
		softargmax_coords = torch.cat([estimated_x, estimated_y], dim=-1)
		return softargmax_coords


class YNet:
	def __init__(self, obs_len, pred_len, params):
		"""
		Ynet class, following a sklearn similar class structure
		:param obs_len: observed timesteps
		:param pred_len: predicted timesteps
		:param params: dictionary with hyperparameters
		"""
		self.obs_len = obs_len
		self.pred_len = pred_len
		self.division_factor = 2 ** len(params['encoder_channels'])

		self.model = YNetTorch(obs_len=obs_len,
							   pred_len=pred_len,
							   segmentation_model_fp=params['segmentation_model_fp'],
							   use_features_only=params['use_features_only'],
							   semantic_classes=params['semantic_classes'],
							   encoder_channels=params['encoder_channels'],
							   decoder_channels=params['decoder_channels'],
							   waypoints=len(params['waypoints']))

	def train(self, train_data, val_data, params, train_image_path, val_image_path, experiment_name, batch_size=8, num_goals=20, num_traj=1, device=None, dataset_name=None, use_raw_data=False, epochs_checkpoints=None, train_net="all", fine_tune=False):
		"""
		Train function
		:param train_data: pd.df, train data
		:param val_data: pd.df, val data
		:param params: dictionary with training hyperparameters
		:param train_image_path: str, filepath to train images
		:param val_image_path: str, filepath to val images
		:param experiment_name: str, arbitrary name to name weights file
		:param batch_size: int, batch size
		:param num_goals: int, number of goals per trajectory, K_e in paper
		:param num_traj: int, number of trajectory per goal, K_a in paper
		:param device: torch.device, if None -> 'cuda' if torch.cuda.is_available() else 'cpu'
		:return:
		"""
		if device is None:
			device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		obs_len = self.obs_len
		pred_len = self.pred_len
		total_len = pred_len + obs_len
		if train_net == "modulator":
			self.model.initialize_style()

		print('Preprocess data')
		dataset_name = dataset_name.lower()
		if dataset_name == 'sdd':
			image_file_name = 'reference.jpg'
		elif dataset_name == 'ind':
			image_file_name = 'reference.png'
		elif dataset_name == 'eth':
			image_file_name = 'oracle.png'
		else:
			raise ValueError(f'{dataset_name} dataset is not supported')

		# ETH/UCY specific: Homography matrix is needed to convert pixel to world coordinates
		if dataset_name == 'eth':
			self.homo_mat = {}
			for scene in ['eth', 'hotel', 'students001', 'students003', 'uni_examples', 'zara1', 'zara2', 'zara3']:
				self.homo_mat[scene] = torch.Tensor(np.loadtxt(f'data/eth_ucy/{scene}_H.txt')).to(device)
			seg_mask = True
		else:
			self.homo_mat = None
			seg_mask = False

		# Load train images and augment train data and images
		df_train, train_images = augment_data(train_data, image_path=train_image_path, image_file=image_file_name,
											  seg_mask=seg_mask, use_raw_data=use_raw_data)

		# Load val scene images
		val_images = create_images_dict(val_data, image_path=val_image_path, image_file=image_file_name, use_raw_data=use_raw_data)

		# Initialize dataloaders
		train_dataset = SceneDataset(df_train, resize=params['resize'], total_len=total_len)
		train_loader = DataLoader(train_dataset, batch_size=1, collate_fn=scene_collate, shuffle=True)

		val_dataset = SceneDataset(val_data, resize=params['resize'], total_len=total_len)
		val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=scene_collate)

		# Preprocess images, in particular resize, pad and normalize as semantic segmentation backbone requires
		resize(train_images, factor=params['resize'], seg_mask=seg_mask)
		pad(train_images, division_factor=self.division_factor)  # make sure that image shape is divisible by 32, for UNet segmentation
		preprocess_image_for_segmentation(train_images, seg_mask=seg_mask)

		resize(val_images, factor=params['resize'], seg_mask=seg_mask)
		pad(val_images, division_factor=self.division_factor)  # make sure that image shape is divisible by 32, for UNet segmentation
		preprocess_image_for_segmentation(val_images, seg_mask=seg_mask)

		model = self.model.to(device)

		# Freeze segmentation model
		for param in model.semantic_segmentation.parameters():
			param.requires_grad = False

		if train_net in ["encoder", "modulator"]:
			for param in model.parameters():
				param.requires_grad = False
			if train_net == "encoder":
				for param in model.encoder.parameters():
					param.requires_grad = True
			elif train_net == "modulator":
				for param in model.style_modulators.parameters():
					param.requires_grad = True


		optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
		criterion = nn.BCEWithLogitsLoss()

		# Create template
		size = int(4200 * params['resize'])

		input_template = create_dist_mat(size=size)
		input_template = torch.Tensor(input_template).to(device)

		gt_template = create_gaussian_heatmap_template(size=size, kernlen=params['kernlen'], nsig=params['nsig'], normalize=False)
		gt_template = torch.Tensor(gt_template).to(device)

		best_test_ADE = 99999999999999

		self.val_ADE = []
		self.val_FDE = []

		with_style = train_net == "modulator"
		print('Start training')
		for e in tqdm(range(params['num_epochs']), desc='Epoch'):
			train_ADE, train_FDE, train_loss = train(model, train_loader, train_images, e, obs_len, pred_len,
													 batch_size, params, gt_template, device,
													 input_template, optimizer, criterion, dataset_name, self.homo_mat, with_style=with_style)

			# For faster inference, we don't use TTST and CWS here, only for the test set evaluation
			val_ADE, val_FDE = evaluate(model, val_loader, val_images, num_goals, num_traj,
										obs_len=obs_len, batch_size=batch_size,
										device=device, input_template=input_template,
										waypoints=params['waypoints'], resize=params['resize'],
										temperature=params['temperature'], use_TTST=False,
										use_CWS=False, dataset_name=dataset_name,
										homo_mat=self.homo_mat, mode='val', with_style=with_style)

			print(f'Epoch {e}: 	Train (Top-1) ADE: {train_ADE:.2f} FDE: {train_FDE:.2f} 		Valid (Top-k) ADE: {val_ADE:.2f} FDE: {val_FDE:.2f}')
			self.val_ADE.append(val_ADE)
			self.val_FDE.append(val_FDE)

			if val_ADE < best_test_ADE:
				best_test_ADE = val_ADE
				best_state_dict = deepcopy(model.state_dict())
				if not fine_tune:
					torch.save(best_state_dict, 'ckpts/' + experiment_name + '_weights.pt')
			
			if e % epochs_checkpoints == 0 and not fine_tune:
				torch.save(model.state_dict(), 'ckpts/' + experiment_name + f'_weights_epoch_{e}.pt')

			# early stop in case of clear overfitting
			if best_test_ADE < min(self.val_ADE[-5:]) * 0.9:
				print(f'Early stop at epoch {e}')
				break

		# Load best model
		model.load_state_dict(best_state_dict, strict=True)
		return self.val_ADE, self.val_FDE

	def evaluate(self, data, params, image_path, batch_size=8, num_goals=20, num_traj=1, rounds=1, device=None, dataset_name=None, use_raw_data=False, with_style=False):
		"""
		Val function
		:param data: pd.df, val data
		:param params: dictionary with training hyperparameters
		:param image_path: str, filepath to val images
		:param batch_size: int, batch size
		:param num_goals: int, number of goals per trajectory, K_e in paper
		:param num_traj: int, number of trajectory per goal, K_a in paper
		:param rounds: int, number of epochs to evaluate
		:param device: torch.device, if None -> 'cuda' if torch.cuda.is_available() else 'cpu'
		:return:
		"""

		if device is None:
			device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
			print('Working on GPU: ', torch.cuda.is_available())

		obs_len = self.obs_len
		pred_len = self.pred_len
		total_len = pred_len + obs_len

		print('Preprocess data')
		dataset_name = dataset_name.lower()
		if dataset_name == 'sdd':
			image_file_name = 'reference.jpg'
		elif dataset_name == 'ind':
			image_file_name = 'reference.png'
		elif dataset_name == 'eth':
			image_file_name = 'oracle.png'
		else:
			raise ValueError(f'{dataset_name} dataset is not supported')

		# ETH/UCY specific: Homography matrix is needed to convert pixel to world coordinates
		if dataset_name == 'eth':
			self.homo_mat = {}
			for scene in ['eth', 'hotel', 'students001', 'students003', 'uni_examples', 'zara1', 'zara2', 'zara3']:
				self.homo_mat[scene] = torch.Tensor(np.loadtxt(f'data/eth_ucy/{scene}_H.txt')).to(device)
			seg_mask = True
		else:
			self.homo_mat = None
			seg_mask = False

		test_images = create_images_dict(data, image_path=image_path, image_file=image_file_name, use_raw_data=use_raw_data)

		test_dataset = SceneDataset(data, resize=params['resize'], total_len=total_len)
		test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=scene_collate)

		# Preprocess images, in particular resize, pad and normalize as semantic segmentation backbone requires
		resize(test_images, factor=params['resize'], seg_mask=seg_mask)
		pad(test_images, division_factor=self.division_factor)  # make sure that image shape is divisible by 32, for UNet architecture
		preprocess_image_for_segmentation(test_images, seg_mask=seg_mask)

		model = self.model.to(device)

		# Create template
		size = int(4200 * params['resize'])

		input_template = torch.Tensor(create_dist_mat(size=size)).to(device)

		self.eval_ADE = []
		self.eval_FDE = []

		print("TTST setting:", params['use_TTST'])
		print('Start testing')
		for e in tqdm(range(rounds), desc='Round'):
			test_ADE, test_FDE = evaluate(model, test_loader, test_images, num_goals, num_traj,
										  obs_len=obs_len, batch_size=batch_size,
										  device=device, input_template=input_template,
										  waypoints=params['waypoints'], resize=params['resize'],
										  temperature=params['temperature'], use_TTST=params['use_TTST'],
										  use_CWS=params['use_CWS'],
										  rel_thresh=params['rel_threshold'], CWS_params=params['CWS_params'],
										  dataset_name=dataset_name, homo_mat=self.homo_mat, mode='test', with_style=with_style)
			print(f'Round {e}: \nTest ADE: {test_ADE} \nTest FDE: {test_FDE}')

			self.eval_ADE.append(test_ADE)
			self.eval_FDE.append(test_FDE)

		ade = sum(self.eval_ADE) / len(self.eval_ADE)
		fde = sum(self.eval_FDE) / len(self.eval_FDE)
		print(f'\n\nAverage performance over {rounds} rounds: \nTest ADE: {ade} \nTest FDE: {fde}')
		return ade, fde


	def load(self, path):
		print(self.model.load_state_dict(torch.load(path)))

	def save(self, path):
		torch.save(self.model.state_dict(), path)
