import numpy as np
import tqdm

from multiprocessing import Pool


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
        image = image.transpose(1, 2, 0)
        H, W, C = image.shape

        ########################################################################
        # UA:
        # ЗАВДАННЯ: Імплементуйте Гістограму Орієнтованих Градієнтів тут.
        #     Дивіться лекцію та посилання на Piazza для деталей.
        #     Функція повинна повертати 1D вектор ознак для заданого зображення.
        ########################################################################
        # EN:
        # TASK: Implement HOG feature extraction here.
        #     See the lecture or the links on Piazza for details.
        #     Must return a 1D vector of features for the given image.
        ########################################################################
        hog_feature = np.zeros((42,))
        # ...

        ########################################################################

        assert len(hog_feature.shape) == 1, hog_feature.shape
        return hog_feature