import torch.utils.data as data
from PIL import Image
import os

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]

def default_loader(path):
    return Image.open(path).convert('RGB')

class ImageFolder(data.Dataset):

    def __init__(self, root_A, root_B, transform=None, return_paths=False, loader=default_loader, max_size=200):
        imgs_A = make_dataset(root_A, max_size)
        imgs_B = make_dataset(root_B, max_size)
        if len(imgs_A) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root_A = root_A
        self.root_B = root_B
        self.imgs_A = imgs_A
        self.imgs_B = imgs_B

        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path_A = self.imgs_A[index]
        path_B = self.imgs_B[index]

        img_A = self.loader(path_A)
        img_B = self.loader(path_B)

        if self.transform is not None:
            img_A = self.transform(img_A)
            img_B = self.transform(img_B)
        if self.return_paths:
            return img_A, img_B, path_A, path_B
        else:
            return img_A, img_B

    def __len__(self):
        return len(self.imgs)