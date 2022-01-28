import torch
import torch.nn as nn
from utils.image_utils import get_patch, image2world
from loss import contrastive_loss
from torch.nn.functional import cross_entropy


def train(model, train_loader, train_images, e, obs_len, pred_len, batch_size, params, gt_template, device, input_template, optimizer, criterion, dataset_name, homo_mat):
	"""
	Run training for one epoch

	:param model: torch model
	:param train_loader: torch dataloader
	:param train_images: dict with keys: scene_name value: preprocessed image as torch.Tensor
	:param e: epoch number
	:param params: dict of hyperparameters
	:param gt_template:  precalculated Gaussian heatmap template as torch.Tensor
	:return: train_ADE, train_FDE, train_loss for one epoch
	"""
	train_loss = 0
	train_ADE = []
	train_FDE = []
	model.train()
	counter = 0
	# outer loop, for loop over each scene as scenes have different image size and to calculate segmentation only once
	for batch, (trajectory, meta, scene) in enumerate(train_loader):
		# Stop training after 25 batches to increase evaluation frequency
		if dataset_name == 'sdd' and obs_len == 8 and batch > 25:
			break

		# TODO Delete
		if dataset_name == 'eth':
			print(counter)
			counter += batch_size
			# Break after certain number of batches to approximate evaluation, else one epoch takes really long
			if counter > 30: #TODO Delete
				break


		# Get scene image and apply semantic segmentation
		if e < params['unfreeze']:  # before unfreeze only need to do semantic segmentation once
			model.eval()
			scene_image = train_images[scene].to(device).unsqueeze(0)
			scene_image = model.segmentation(scene_image)
			model.train()

		# inner loop, for each trajectory in the scene
		for i in range(0, len(trajectory), batch_size):
			if e >= params['unfreeze']:
				scene_image = train_images[scene].to(device).unsqueeze(0)
				scene_image = model.segmentation(scene_image)

			# Create Heatmaps for past and ground-truth future trajectories
			_, _, H, W = scene_image.shape  # image shape

			observed = trajectory[i:i+batch_size, :obs_len, :].reshape(-1, 2).cpu().numpy()
			observed_map = get_patch(input_template, observed, H, W)
			observed_map = torch.stack(observed_map).reshape([-1, obs_len, H, W])

			gt_future = trajectory[i:i + batch_size, obs_len:].to(device)
			gt_future_map = get_patch(gt_template, gt_future.reshape(-1, 2).cpu().numpy(), H, W)
			gt_future_map = torch.stack(gt_future_map).reshape([-1, pred_len, H, W])

			gt_waypoints = gt_future[:, params['waypoints']]
			gt_waypoint_map = get_patch(input_template, gt_waypoints.reshape(-1, 2).cpu().numpy(), H, W)
			gt_waypoint_map = torch.stack(gt_waypoint_map).reshape([-1, gt_waypoints.shape[1], H, W])

			# Concatenate heatmap and semantic map
			semantic_map = scene_image.expand(observed_map.shape[0], -1, -1, -1)  # expand to match heatmap size
			feature_input = torch.cat([semantic_map, observed_map], dim=1)

			# Forward pass
			# Calculate features
			features = model.pred_features(feature_input)

			# Predict goal and waypoint probability distribution
			pred_goal_map = model.pred_goal(features)
			goal_loss = criterion(pred_goal_map, gt_future_map) * params['loss_scale']  # BCEWithLogitsLoss

			# Prepare (downsample) ground-truth goal and trajectory heatmap representation for conditioning trajectory decoder
			gt_waypoints_maps_downsampled = [nn.AvgPool2d(kernel_size=2**i, stride=2**i)(gt_waypoint_map) for i in range(1, len(features))]
			gt_waypoints_maps_downsampled = [gt_waypoint_map] + gt_waypoints_maps_downsampled

			# Predict trajectory distribution conditioned on goal and waypoints
			traj_input = [torch.cat([feature, goal], dim=1) for feature, goal in zip(features, gt_waypoints_maps_downsampled)]
			pred_traj_map = model.pred_traj(traj_input)
			traj_loss = criterion(pred_traj_map, gt_future_map) * params['loss_scale']  # BCEWithLogitsLoss

			# Backprop
			loss = goal_loss + traj_loss
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			with torch.no_grad():
				train_loss += loss
				# Evaluate using Softargmax, not a very exact evaluation but a lot faster than full prediction
				pred_traj = model.softargmax(pred_traj_map)
				pred_goal = model.softargmax(pred_goal_map[:, -1:])

				# converts ETH/UCY pixel coordinates back into world-coordinates
				# if dataset_name == 'eth':
				# 	pred_goal = image2world(pred_goal, scene, homo_mat, params)
				# 	pred_traj = image2world(pred_traj, scene, homo_mat, params)
				# 	gt_future = image2world(gt_future, scene, homo_mat, params)

				train_ADE.append(((((gt_future - pred_traj) / params['resize']) ** 2).sum(dim=2) ** 0.5).mean(dim=1))
				train_FDE.append(((((gt_future[:, -1:] - pred_goal[:, -1:]) / params['resize']) ** 2).sum(dim=2) ** 0.5).mean(dim=1))

	train_ADE = torch.cat(train_ADE).mean()
	train_FDE = torch.cat(train_FDE).mean()

	return train_ADE.item(), train_FDE.item(), train_loss.item()


def train_style_enc(model, train_loaders, train_images, e, obs_len, pred_len, batch_size, params, gt_template, device, input_template, optimizer, criterion, dataset_name, homo_mat, style_only=True):
	"""
	Run training for one epoch

	:param model: torch model
	:param train_loader: torch dataloaders
	:param train_images: dict with keys: scene_name value: preprocessed image as torch.Tensor
	:param e: epoch number
	:param params: dict of hyperparameters
	:param gt_template:  precalculated Gaussian heatmap template as torch.Tensor
	:return: train_ADE, train_FDE, train_loss for one epoch
	"""
	train_loss, train_accuracy = [], []
	model.train()
	batch = 0

	while batch < len(train_loaders[0]):

		batch += 1
		loss = 0
		scene_images, trajectories, scenes = []

		for i, train_loader in enumerate(train_loaders):

			(trajectory, meta, scene) = next(train_loader)
			trajectories.append(trajectory)
			scenes.append(scenes)

			# Stop training after 25 batches to increase evaluation frequency
			if dataset_name == 'sdd' and obs_len == 8 and batch > 25:				# TODO: why's it?
				return torch.cat(train_loss).mean().item(), torch.cat(train_accuracy).mean().item()

			# Get scene image and apply semantic segmentation
			if e < params['unfreeze']:  # before unfreeze only need to do semantic segmentation once
				model.eval()
				scene_image = train_images[scene].to(device).unsqueeze(0)
				scene_images.append(model.segmentation(scene_image))
				model.train()

		len_min_trajectories = min([len(traj) for traj in trajectories])

		for i in range(0, len_min_trajectories, batch_size):

			low_dim_style_list = []
			classified_style_list = []
			loss = 0
			
			for scene_image, scene, trajectory in zip(scene_images, scenes, trajectories):

				if e >= params['unfreeze']:
					scene_image = train_images[scene].to(device).unsqueeze(0)
					scene_image = model.segmentation(scene_image)

				# Create Heatmaps for past and ground-truth future trajectories
				_, _, H, W = scene_image.shape  # image shape

				observed_style = trajectory[i:i+batch_size, :, :].reshape(-1, 2).cpu().numpy() # NO OBS LENGTH TO KEEP ENTIRE OBSERVATION 
				observed_map_style = get_patch(input_template, observed_style, H, W)
				observed_map_style = torch.stack(observed_map_style).reshape([-1, trajectory.shape[1], H, W])

				# Concatenate heatmap and semantic map
				semantic_map = scene_image.expand(observed_map_style.shape[0], -1, -1, -1)  # expand to match heatmap size
				feature_input = torch.cat([semantic_map, observed_map_style], dim=1)

				# Forward pass
				style_features = model.style_features(feature_input)
				low_dim_style_features = model.style_low_dim(style_features)
				low_dim_style_list.append(low_dim_style_features)

				# For classification
				class_features = model.style_class_dim(low_dim_style_features).detach()
				classified_style_list.append(class_features)

				# Add normal loss if also training the rest
				if not style_only:

					observed = trajectory[i:i+batch_size, :obs_len, :].reshape(-1, 2).cpu().numpy() # NO OBS LENGTH TO KEEP ENTIRE OBSERVATION 
					observed_map = get_patch(input_template, observed, H, W)
					observed_map = torch.stack(observed_map).reshape([-1, obs_len, H, W])

					gt_future = trajectory[i:i + batch_size, obs_len:].to(device)
					gt_future_map = get_patch(gt_template, gt_future.reshape(-1, 2).cpu().numpy(), H, W)
					gt_future_map = torch.stack(gt_future_map).reshape([-1, pred_len, H, W])

					gt_waypoints = gt_future[:, params['waypoints']]
					gt_waypoint_map = get_patch(input_template, gt_waypoints.reshape(-1, 2).cpu().numpy(), H, W)
					gt_waypoint_map = torch.stack(gt_waypoint_map).reshape([-1, gt_waypoints.shape[1], H, W])

					# Concatenate heatmap and semantic map
					semantic_map = scene_image.expand(observed_map.shape[0], -1, -1, -1)  # expand to match heatmap size
					feature_input = torch.cat([semantic_map, observed_map], dim=1)

					features = model.pred_features(feature_input) 

					# Predict goal and waypoint probability distribution
					pred_goal_map = model.pred_goal(features)
					goal_loss = criterion(pred_goal_map, gt_future_map) * params['loss_scale']  # BCEWithLogitsLoss

					# Prepare (downsample) ground-truth goal and trajectory heatmap representation for conditioning trajectory decoder
					gt_waypoints_maps_downsampled = [nn.AvgPool2d(kernel_size=2**i, stride=2**i)(gt_waypoint_map) for i in range(1, len(features))]
					gt_waypoints_maps_downsampled = [gt_waypoint_map] + gt_waypoints_maps_downsampled

					# Predict trajectory distribution conditioned on goal and waypoints
					traj_input = [torch.cat([feature, goal], dim=1) for feature, goal in zip(features, gt_waypoints_maps_downsampled)]
					pred_traj_map = model.pred_traj(traj_input)
					traj_loss = criterion(pred_traj_map, gt_future_map) * params['loss_scale']  # BCEWithLogitsLoss

					# Backprop
					loss += goal_loss + traj_loss


			# Contrastive loss
			input_features = torch.stack(low_dim_style_list)
			input_labels = torch.range(len(train_loaders)).unsqueeze().to(device)
			loss += contrastive_loss(input_features, input_labels) * params['contrast_loss_scale']

			# Classifier
			classifier_features = torch.cat(classified_style_list)
			classifier_labels = torch.stack([
				torch.ones(feat.shape[0]) * i
			] for i, feat in classified_style_list).to(device)
			loss += cross_entropy(classifier_features, classifier_labels)

			# Backpropagate 
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			# Metrics
			classifier_prediction = classifier_features.argmax(dim=-1)
			accuracy = ((classifier_prediction == classifier_labels) * 1).to(float).mean()
			train_accuracy.append(accuracy.detach())
			train_loss.append(loss.detach())

	return torch.cat(train_loss).mean().item(), torch.cat(train_accuracy).mean().item()
