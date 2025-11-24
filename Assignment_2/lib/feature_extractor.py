import numpy as np
import tqdm

from multiprocessing import Pool

import cv2

class FeatureExtractor:
    def __init__(self, use_multiprocessing=True):
        self.feature_dim = None
        self.use_multiprocessing = use_multiprocessing

    def extract(self, images):
        num_images = images.shape[0]
        feature0 = self.extract_from_image(images[0])
        feature_dim = feature0.shape[0]

        assert self.feature_dim == feature_dim or self.feature_dim is None, "Feature size changes!"
        self.feature_dim = feature_dim

        computed_features = np.empty((num_images, self.feature_dim))
        computed_features[0] = feature0

        if self.use_multiprocessing:
            with Pool() as p:
                result_iterator = p.imap(self.extract_from_image, images[1:])
                for i, comp_feature_vec in enumerate(tqdm.tqdm(result_iterator)):
                    computed_features[i+1] = comp_feature_vec
        else:
            for i in tqdm.tqdm(range(1, images.shape[0])):
                computed_features[i] = self.extract_from_image(images[i])

        return computed_features

    def extract_from_image(self, image):
        raise NotImplementedError


class BaselineFeatureExtractor(FeatureExtractor):
    """
    A simple feature extractor which computes mean and std of
    4x4 blocks of pixels and concatenates the results together.
    """
    def extract_from_image(self, image):
        C, H, W = image.shape
        image = image.transpose(1, 2, 0)
        H, W, C = image.shape

        # to [-1, 1]
        image = image / 127.5 - 1.0
        means = []
        stds = []

        for i in range(H // 4):
            for j in range(W // 4):
                block = image[i:i+4, j:j+4]
                block_mean = np.mean(block, axis=(0, 1))
                block_std = np.std(block, axis=(0, 1))
                means.append(block_mean)
                stds.append(block_std)
        
        feature_vector = np.concatenate(means + stds)
        return feature_vector  # feature dim = 4*4*3*2 = 96


class FeatureExtractor:
    def __init__(self, use_multiprocessing=True):
        self.feature_dim = None
        self.use_multiprocessing = use_multiprocessing

    def extract(self, images):
        num_images = images.shape[0]
        feature0 = self.extract_from_image(images[0])
        feature_dim = feature0.shape[0]

        assert self.feature_dim == feature_dim or self.feature_dim is None, "Feature size changes!"
        self.feature_dim = feature_dim

        computed_features = np.empty((num_images, self.feature_dim))
        computed_features[0] = feature0

        if self.use_multiprocessing:
            with Pool() as p:
                result_iterator = p.imap(self.extract_from_image, images[1:])
                for i, comp_feature_vec in enumerate(tqdm.tqdm(result_iterator)):
                    computed_features[i+1] = comp_feature_vec
        else:
            for i in tqdm.tqdm(range(1, images.shape[0])):
                computed_features[i] = self.extract_from_image(images[i])

        return computed_features

    def extract_from_image(self, image):
        raise NotImplementedError


class BaselineFeatureExtractor(FeatureExtractor):
    """
    A simple feature extractor which computes mean and std of
    4x4 blocks of pixels and concatenates the results together.
    """
    def extract_from_image(self, image):
        C, H, W = image.shape
        image = image.transpose(1, 2, 0)
        H, W, C = image.shape

        # to [-1, 1]
        image = image / 127.5 - 1.0
        means = []
        stds = []

        for i in range(H // 4):
            for j in range(W // 4):
                block = image[i:i+4, j:j+4]
                block_mean = np.mean(block, axis=(0, 1))
                block_std = np.std(block, axis=(0, 1))
                means.append(block_mean)
                stds.append(block_std)
        
        feature_vector = np.concatenate(means + stds)
        return feature_vector  # feature dim = 4*4*3*2 = 96


class HOGFeatureExtractor(FeatureExtractor):
    def extract_from_image(self, image):
        # image shape is (C, H, W) -> transpose to (H, W, C)
        image = image.transpose(1, 2, 0)
        H, W, C = image.shape

        # 1. Gamma Correction (Square Root Compression)
        image = np.sqrt(image)
        
        # Ensure image is float32 for OpenCV
        image = image.astype(np.float32)

        # 2. Gaussian Smoothing
        # Use a small kernel (e.g., 3x3)
        # We need to handle both 3-channel and 1-channel images
        if C == 3:
            # cv2.GaussianBlur expects uint8 or float32.
            image = cv2.GaussianBlur(image, (3, 3), 0)
        else:
            # For 1 channel, we might need to squeeze/expand or just pass it
            image = cv2.GaussianBlur(image, (3, 3), 0)
            if len(image.shape) == 2:
                image = image[:, :, np.newaxis]

        # 3. Compute gradients
        # We compute gradients for ALL channels and pick the max magnitude
        
        # Pad image to handle borders
        image_pad = np.pad(image, ((1, 1), (1, 1), (0, 0)), mode='edge')
        
        # Gradients for each channel
        # shape: (H, W, C)
        gx = image_pad[1:-1, 2:, :] - image_pad[1:-1, :-2, :]
        gy = image_pad[2:, 1:-1, :] - image_pad[:-2, 1:-1, :]
        
        # Magnitude and Orientation for each channel
        mag_channels = np.sqrt(gx**2 + gy**2)
        ori_channels = np.arctan2(gy, gx) * (180 / np.pi) % 180
        
        # Find max magnitude across channels
        # shape: (H, W)
        max_mag_idx = np.argmax(mag_channels, axis=2)
        
        # Select the magnitude and orientation corresponding to the max magnitude
        # We use advanced indexing
        y_grid, x_grid = np.indices((H, W))
        magnitude = mag_channels[y_grid, x_grid, max_mag_idx]
        orientation = ori_channels[y_grid, x_grid, max_mag_idx]

        # 4. Compute Cell Histograms
        # Reduced cell size for better detail
        cell_size = 4
        n_bins = 9
        
        n_cells_y = H // cell_size
        n_cells_x = W // cell_size
        
        histograms = np.zeros((n_cells_y, n_cells_x, n_bins))
        
        for y in range(n_cells_y):
            for x in range(n_cells_x):
                cell_mag = magnitude[y*cell_size:(y+1)*cell_size, x*cell_size:(x+1)*cell_size]
                cell_ori = orientation[y*cell_size:(y+1)*cell_size, x*cell_size:(x+1)*cell_size]
                
                # Histogram computation
                hist, _ = np.histogram(cell_ori, bins=n_bins, range=(0, 180), weights=cell_mag)
                histograms[y, x, :] = hist

        # 5. Block Normalization
        # Block size 2x2 cells
        block_size = 2
        n_blocks_y = n_cells_y - block_size + 1
        n_blocks_x = n_cells_x - block_size + 1
        
        normalized_blocks = []
        
        epsilon = 1e-5
        
        for y in range(n_blocks_y):
            for x in range(n_blocks_x):
                block = histograms[y:y+block_size, x:x+block_size, :]
                # Flatten block
                block_feat = block.flatten()
                # L2 norm
                norm = np.sqrt(np.sum(block_feat**2) + epsilon)
                normalized_blocks.append(block_feat / norm)
                
        hog_feature = np.concatenate(normalized_blocks)

        assert len(hog_feature.shape) == 1, hog_feature.shape
        return hog_feature