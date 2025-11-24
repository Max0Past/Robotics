import os
import numpy as np
import pickle
import urllib.request
import tarfile

ROOT_DIR = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), os.pardir)))

class CIFAR10:
    """
    Implemented as a class just for clarity. No actual state exists. Don't do like that in the wild :D
    """

    IMAGE_SHAPE = (3, 32, 32)
    TAR = "cifar-10-python.tar.gz"
    URL = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

    NUM_VAL = 1000
    NUM_TEST = 1000
    NUM_TRAIN = 49000

    LABEL_NAMES = [
        "airplane", "automobile", "bird",
        "cat", "deer", "dog", "frog",
        "horse", "ship", "truck",
    ]

    def __init__(self):
        self.root = ROOT_DIR
        self.dataset_path = os.path.join(self.root, "cifar-10-batches-py")
        self.tar_path = os.path.join(self.root, CIFAR10.TAR)

        self._try_download()

    def load_dataset(self):
        def _read_chunk(file_path):
            # NOTE: CIFAR10 dataset uses the term "batch". We use "chunk"
            #       instead to avoid confusion with "training batch".
            with open(file_path, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
                images = data['data']
                labels = data['labels']
                images = images.reshape(-1, *CIFAR10.IMAGE_SHAPE)
                labels = np.array(labels)
                return images, labels

        image_chunks = []
        label_chunks = []
        for batch in range(1, 5 + 1):
            file_path = os.path.join(self.dataset_path, f'data_batch_{batch}')
            image_chunk, label_chunk = _read_chunk(file_path)
            image_chunks.append(image_chunk)
            label_chunks.append(label_chunk)
        trainval_images = np.concatenate(image_chunks)
        trainval_labels = np.concatenate(label_chunks)
        test_images, test_labels = _read_chunk(os.path.join(self.dataset_path, 'test_batch'))

        train_images = trainval_images[:CIFAR10.NUM_TRAIN]
        train_labels = trainval_labels[:CIFAR10.NUM_TRAIN]
        val_images = trainval_images[CIFAR10.NUM_TRAIN:]
        val_labels = trainval_labels[CIFAR10.NUM_TRAIN:]
        test_images = test_images[:CIFAR10.NUM_TEST]
        test_labels = test_labels[:CIFAR10.NUM_TEST]

        return train_images, train_labels, val_images, val_labels, test_images, test_labels
    
    def _try_download(self):
        if os.path.exists(self.dataset_path):
            print("CIFAR10 found.")
            return
        
        print("Downloading CIFAR10...")
        if not os.path.exists(self.tar_path):
            # Use urllib instead of wget for cross-platform compatibility
            print(f"Downloading from {CIFAR10.URL}...")
            urllib.request.urlretrieve(CIFAR10.URL, self.tar_path)
            print("Download complete.")

        print("Unpacking CIFAR10...")
        # Use tarfile module instead of tar command
        with tarfile.open(self.tar_path, 'r:gz') as tar:
            tar.extractall(path=self.root)
        print("Extraction complete.")

        print("Cleaning up...")
        os.remove(self.tar_path)

        print("CIFAR10 downloaded successfully")