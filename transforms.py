import numpy as np
import os
import sys
from skimage import filters, exposure
import matplotlib.pyplot as plt
from PIL import Image
import torch

class Normalize(object):
    """
    Normalize an image using mean and standard deviation.
    
    Args:
        mean (float or tuple): Mean for each channel.
        std (float or tuple): Standard deviation for each channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        """
        Args:
            data (tuple): Containing input and target images to be normalized.
        
        Returns:
            Tuple: Normalized input and target images.
        """
        input_img, target_img = data

        # Normalize input image
        input_normalized = (input_img - self.mean) / self.std

        # Normalize target image
        target_normalized = (target_img - self.mean) / self.std

        return input_normalized, target_normalized


class RandomFlip(object):

    def __call__(self, data):

        input_img, target_img = data

        if np.random.rand() > 0.5:
            input_img = np.fliplr(input_img)
            target_img = np.fliplr(target_img)

        if np.random.rand() > 0.5:
            input_img = np.flipud(input_img)
            target_img = np.flipud(target_img)

        return input_img, target_img


class BatchRandomFlip(object):
    """
    Randomly flip patches in a batch of images horizontally and/or vertically.
    Works with batches of images in the format (batch_size, height, width, channels).
    """

    def __call__(self, batch_data):
        """
        Args:
            batch_data (tuple): A tuple containing two numpy arrays (input_patches, target_patches),
                                each with the shape (batch_size, height, width, channels).
        
        Returns:
            Tuple: Two numpy arrays with flipped images in the same shape as input.
        """
        input_patches, target_patches = batch_data
        batch_size = input_patches.shape[0]

        for i in range(batch_size):
            if np.random.rand() > 0.5:
                # Flip patches horizontally
                input_patches[i] = np.fliplr(input_patches[i])
                target_patches[i] = np.fliplr(target_patches[i])

            if np.random.rand() > 0.5:
                # Flip patches vertically
                input_patches[i] = np.flipud(input_patches[i])
                target_patches[i] = np.flipud(target_patches[i])

        return input_patches, target_patches



class RandomCrop(object):

    def __init__(self, output_size=(64,64)):
        self.output_size = output_size

    def __call__(self, data):

        h, w = data[0].shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        id_y = np.arange(top, top + new_h, 1)[:, np.newaxis].astype(np.int32)
        id_x = np.arange(left, left + new_w, 1).astype(np.int32)

        return data[0][id_y, id_x],data[1][id_y, id_x]
    

class RandomCropVideo(object):
    """
    Randomly crop images from a stack of grayscale slices (input) and a single grayscale slice (target),
    where each slice includes a channel dimension.
    """

    def __init__(self, output_size=(64, 64)):
        """
        Initializes the RandomCrop transformer with the desired output size.

        Parameters:
        - output_size (tuple): The target output size (height, width).
        """
        self.output_size = output_size

    def __call__(self, data):
        """
        Apply random cropping to a stack of input slices and a single target slice.

        Parameters:
        - data (tuple): A tuple containing the input stack and target slice.

        Returns:
        - Tuple: Randomly cropped input stack and target slice.
        """
        input_stack, target_slice = data
        h, w = input_stack.shape[1:3]  # Assuming input_stack shape is (num_slices, H, W, C)
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)

        # Crop each slice in the input stack
        cropped_input_stack = input_stack[:, top:top+new_h, left:left+new_w, :]

        # Crop the target slice
        cropped_target_slice = target_slice[top:top+new_h, left:left+new_w, :]

        return cropped_input_stack, cropped_target_slice



class GenerateFixedPatches(object):
    """
    Generate 8 patches of size 64x64 from the input and target images.
    The images are first padded if they are smaller than 64x64.
    Patches are selected to cover the image as evenly as possible.
    """

    def __init__(self, patch_size=(80, 80), num_patches=8):
        self.patch_size = patch_size
        self.num_patches = num_patches

    def __call__(self, data):
        input_img, target_img = data
        h, w = input_img.shape[:2]
        new_h, new_w = self.patch_size

        # Ensure the image is at least 64x64 by padding if necessary
        pad_h = max(0, new_h - h)
        pad_w = max(0, new_w - w)
        if pad_h > 0 or pad_w > 0:
            input_img = np.pad(input_img, ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2), (0, 0)), mode='constant')
            target_img = np.pad(target_img, ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2), (0, 0)), mode='constant')
            h, w = input_img.shape[:2]

        patches_input = []
        patches_target = []

        # Calculate step sizes to try to distribute patches evenly across the image
        step_h = max(1, (h - new_h) // (self.num_patches // 2))
        step_w = max(1, (w - new_w) // (self.num_patches // 2))

        count = 0
        for i in range(0, h - new_h + 1, step_h):
            for j in range(0, w - new_w + 1, step_w):
                if count < self.num_patches:
                    patch_input = input_img[i:i+new_h, j:j+new_w, :]
                    patch_target = target_img[i:i+new_h, j:j+new_w, :]
                    patches_input.append(patch_input)
                    patches_target.append(patch_target)
                    count += 1
                else:
                    break
            if count >= self.num_patches:
                break

        # Convert lists to numpy arrays with an additional dimension for batch size
        patches_input = np.stack(patches_input, axis=0)
        patches_target = np.stack(patches_target, axis=0)

        return patches_input, patches_target


class GenerateRandomPatches(object):
    """
    Generate 8 patches of size 80x80 from the input and target images.
    The images are first padded if they are smaller than 80x80.
    Patches are selected at random positions within the image.
    """

    def __init__(self, patch_size=(128, 128), num_patches=8):
        self.patch_size = patch_size
        self.num_patches = num_patches

    def __call__(self, data):
        input_img, target_img = data
        h, w, _ = input_img.shape
        new_h, new_w = self.patch_size

        # Ensure the image is at least 80x80 by padding if necessary
        pad_h = max(0, new_h - h)
        pad_w = max(0, new_w - w)
        if pad_h > 0 or pad_w > 0:
            input_img = np.pad(input_img, ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2), (0, 0)), mode='constant', constant_values=0)
            target_img = np.pad(target_img, ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2), (0, 0)), mode='constant', constant_values=0)
            h, w = input_img.shape[:2]

        patches_input = []
        patches_target = []

        for _ in range(self.num_patches):
            # Randomly select the top-left corner of the patch
            i = np.random.randint(0, h - new_h + 1)
            j = np.random.randint(0, w - new_w + 1)

            patch_input = input_img[i:i+new_h, j:j+new_w, :]
            patch_target = target_img[i:i+new_h, j:j+new_w, :]
            patches_input.append(patch_input)
            patches_target.append(patch_target)

        # Convert lists to numpy arrays with an additional dimension for batch size
        patches_input = np.stack(patches_input, axis=0)
        patches_target = np.stack(patches_target, axis=0)

        return patches_input, patches_target


class GenerateEquidistantIntensityPatches(object):
    def __init__(self, patch_size=(128, 128), total_patches=64, select_patches=8):
        self.patch_size = patch_size
        self.total_patches = total_patches
        self.select_patches = select_patches

    def __call__(self, data):
        input_img, target_img = data
        h, w, _ = input_img.shape
        new_h, new_w = self.patch_size

        # Ensure the image is at least the size of patch_size by padding if necessary
        pad_h = max(0, new_h - h)
        pad_w = max(0, new_w - w)
        input_img = np.pad(input_img, ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2), (0, 0)), mode='constant', constant_values=0)
        target_img = np.pad(target_img, ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2), (0, 0)), mode='constant', constant_values=0)

        patches_input = []
        patches_target = []
        intensities = []

        # Generate more patches than needed
        while len(patches_input) < self.total_patches:
            i = np.random.randint(0, h - new_h + 1)
            j = np.random.randint(0, w - new_w + 1)

            patch_input = input_img[i:i+new_h, j:j+new_w, :]
            patch_target = target_img[i:i+new_h, j:j+new_w, :]
            patches_input.append(patch_input)
            patches_target.append(patch_target)
            # Calculate and store the average intensity of the input patch
            intensities.append(np.mean(patch_input))

        # Order patches by intensity
        ordered_indices = np.argsort(intensities)
        selected_indices = np.linspace(0, len(ordered_indices) - 1, self.select_patches, dtype=int)

        # Select equidistant patches
        selected_input_patches = np.array(patches_input)[ordered_indices][selected_indices]
        selected_target_patches = np.array(patches_target)[ordered_indices][selected_indices]

        return selected_input_patches, selected_target_patches



class GenerateRandomPatchesIntensity(object):
    """
    Generate patches of specified size from the input and target images.
    The images are first padded if they are smaller than the specified patch size.
    Patches are selected at random positions within the image, with a minimum average intensity threshold.
    If suitable patches are not found after a certain number of attempts, it defaults to selecting random patches.
    """

    def __init__(self, patch_size=(128, 128), num_patches=8, intensity_threshold=0.1, max_attempts=100):
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.intensity_threshold = intensity_threshold
        self.max_attempts = max_attempts

    def __call__(self, data):
        input_img, target_img = data
        h, w, _ = input_img.shape
        new_h, new_w = self.patch_size

        # Ensure the image is at least the size of patch_size by padding if necessary
        pad_h = max(0, new_h - h)
        pad_w = max(0, new_w - w)
        if pad_h > 0 or pad_w > 0:
            input_img = np.pad(input_img, ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2), (0, 0)), mode='constant', constant_values=0)
            target_img = np.pad(target_img, ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2), (0, 0)), mode='constant', constant_values=0)

        patches_input = []
        patches_target = []
        attempts = 0

        while len(patches_input) < self.num_patches and attempts < self.max_attempts:
            # Randomly select the top-left corner of the patch
            i = np.random.randint(0, h - new_h + 1)
            j = np.random.randint(0, w - new_w + 1)

            patch_input = input_img[i:i+new_h, j:j+new_w, :]
            patch_target = target_img[i:i+new_h, j:j+new_w, :]

            # Check if the patch meets the intensity threshold criteria
            if np.mean(patch_input) >= self.intensity_threshold or attempts >= self.max_attempts - self.num_patches:
                patches_input.append(patch_input)
                patches_target.append(patch_target)
            attempts += 1

        # If not enough patches meet the criteria, fill in with the last attempted patches to ensure num_patches are returned
        while len(patches_input) < self.num_patches:
            patches_input.append(patch_input)
            patches_target.append(patch_target)

        # Convert lists to numpy arrays with an additional dimension for batch size
        patches_input = np.stack(patches_input, axis=0)
        patches_target = np.stack(patches_target, axis=0)

        return patches_input, patches_target





class CropToMultipleOf16Inference(object):
    """
    Crop an image to ensure its height and width are multiples of 16.
    This is particularly useful for models that require input dimensions to be divisible by certain values.
    """

    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray): Image to be cropped.

        Returns:
            numpy.ndarray: Cropped image.
        """
        h, w = img.shape[:2]  # Assuming img is a numpy array with shape (H, W, C) or (H, W)

        # Compute new dimensions to be multiples of 16
        new_h = h - (h % 16)
        new_w = w - (w % 16)

        # Calculate cropping margins
        top = (h - new_h) // 2
        left = (w - new_w) // 2

        # Generate indices for cropping
        id_y = np.arange(top, top + new_h, 1)[:, np.newaxis].astype(np.int32)
        id_x = np.arange(left, left + new_w, 1).astype(np.int32)

        # Crop the image
        cropped_image = img[id_y, id_x]

        return cropped_image


class CropToMultipleOf16Video(object):
    """
    Crop a stack of images and a single image to ensure their height and width are multiples of 16.
    This is particularly useful for models that require input dimensions to be divisible by certain values.
    """

    def __call__(self, data):
        """
        Args:
            data (tuple): A tuple where the first element is an input stack with shape (4, H, W, 1)
                          and the second element is a target image with shape (H, W, 1).

        Returns:
            tuple: A tuple containing the cropped input stack and target image.
        """
        input_stack, target_img = data

        # Crop the input stack
        cropped_input_stack = [self.crop_image(frame) for frame in input_stack]

        # Crop the target image
        cropped_target_img = self.crop_image(target_img)

        return np.array(cropped_input_stack), cropped_target_img

    def crop_image(self, img):
        """
        Crop a single image to make its dimensions multiples of 16.

        Args:
            img (numpy.ndarray): Single image to be cropped.

        Returns:
            numpy.ndarray: Cropped image.
        """
        h, w = img.shape[:2]  # Assuming img is a numpy array with shape (H, W, C) or (H, W)

        # Compute new dimensions to be multiples of 16
        new_h = h - (h % 16)
        new_w = w - (w % 16)

        # Calculate cropping margins
        top = (h - new_h) // 2
        left = (w - new_w) // 2

        # Crop the image
        cropped_image = img[top:top+new_h, left:left+new_w]

        return cropped_image



class SquareCrop(object):
    def __call__(self, data):
        """
        Crop input and target images to form square images.

        Args:
        - data: a tuple containing input and target images as NumPy arrays.

        Returns:
        - A tuple containing cropped input and target images as NumPy arrays.
        """
        input_img, target_img = data

        # Get the minimum side length
        min_side_length = min(input_img.shape[0], input_img.shape[1])

        # Calculate the cropping boundaries
        start_row = (input_img.shape[0] - min_side_length) // 2
        start_col = (input_img.shape[1] - min_side_length) // 2

        # Crop input image
        cropped_input = input_img[start_row:start_row + min_side_length,
                                   start_col:start_col + min_side_length, :]

        # Crop target image
        cropped_target = target_img[start_row:start_row + min_side_length,
                                     start_col:start_col + min_side_length, :]

        return cropped_input, cropped_target


class ZeroPad(object):

    def __init__(self, new_size=256):

        self.new_size = new_size

    def __call__(self, data):

        input_img, target_img = data

        h, w = input_img.shape[-3:-1]

        l = (self.new_size - w)//2
        r = (self.new_size - w) - l

        u = (self.new_size - h)//2
        b = (self.new_size - h) - u

        input_img = np.pad(input_img, pad_width=((u, b), (l, r), (0, 0)))
        target_img = np.pad(target_img, pad_width=((u, b), (l, r), (0, 0)))

        return input_img, target_img



class ToTensor(object):
    """
    Convert images or batches of images to PyTorch tensors, handling both single images
    and tuples of images (input_img, target_img). The input is expected to be in the format
    (b, h, w, c) for batches or (h, w, c) for single images, and it converts them to
    PyTorch's (b, c, h, w) format or (c, h, w) for single images.
    """

    def __call__(self, data):
        """
        Convert input images or a tuple of images to PyTorch tensors, adjusting the channel position.

        Args:
            data (numpy.ndarray or tuple of numpy.ndarray): The input can be a single image (h, w, c),
            a batch of images (b, h, w, c), or a tuple of (input_img, target_img) in similar formats.

        Returns:
            torch.Tensor or tuple of torch.Tensor: The converted image(s) as PyTorch tensor(s) in the
            format (c, h, w) for single images or (b, c, h, w) for batches. If input is a tuple, returns
            a tuple of tensors.
        """
        def convert_image(img):
            # Convert a single image or a batch of images to a tensor, adjusting channel position
            if img.ndim == 4:  # Batch of images (b, h, w, c)
                return torch.from_numpy(img.transpose(0, 3, 1, 2).astype(np.float32))
            elif img.ndim == 3:  # Single image (h, w, c)
                return torch.from_numpy(img.transpose(2, 0, 1).astype(np.float32))
            else:
                raise ValueError("Unsupported image format: must be (h, w, c) or (b, h, w, c).")

        # Check if the input is a tuple of images
        if isinstance(data, tuple):
            return tuple(convert_image(img) for img in data)
        else:
            return convert_image(data)



class ToTensorVideo(object):
    """
    Convert stacks of images or single images to PyTorch tensors, specifically handling a tuple
    of a stack of input frames and a single target frame for grayscale images.
    The input stack is expected to have the shape (4, H, W, 1) and the target image (H, W, 1).
    It converts them to PyTorch's (B, C, H, W) format for the stack and (C, H, W) for the single image.
    """

    def __call__(self, data):
        """
        Convert a tuple of input stack and target image to PyTorch tensors, adjusting the channel position.
        
        Args:
            data (tuple): A tuple where the first element is an input stack with shape (4, H, W, 1)
                          and the second element is a target image with shape (H, W, 1).
        
        Returns:
            tuple of torch.Tensor: A tuple containing the converted input stack as a PyTorch tensor
                                   with shape (4, 1, H, W) and the target image as a PyTorch tensor
                                   with shape (1, H, W).
        """
        input_stack, target_img = data
        
        # Convert the input stack to tensor and adjust dimensions
        input_stack_tensor = torch.from_numpy(input_stack.transpose(3, 1, 2, 0).astype(np.float32))
        
        # Convert the target image to tensor and adjust dimensions
        target_img_tensor = torch.from_numpy(target_img.transpose(2, 0, 1).astype(np.float32))
        
        return input_stack_tensor, target_img_tensor



class ToNumpy(object):

    def __call__(self, data):

        return data.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
    
    
class ToNumpyVideo(object):
    """
    Convert PyTorch tensors to numpy arrays, handling single images, batches of images, 
    and stacks of images separately. Adjusts dimensions to move the channel dimension to the last.
    """
    def __call__(self, tensor):
        # Convert a single image or a batch of images to a numpy array
        if tensor.ndim == 4:  # Single image or batch of images (B, C, H, W)
            return tensor.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
        elif tensor.ndim == 5:  # Stack of images (B, C, H, W, Num_Frames)
            return tensor.to('cpu').detach().numpy().transpose(0, 2, 3, 1, 4)
        else:
            raise ValueError("Unsupported tensor format: input must be a single image (C, H, W), \
                             a batch of images (B, C, H, W), or a stack of images (B, C, H, W, Num_Frames).")





class Denormalize(object):
    """
    Denormalize an image using mean and standard deviation.
    
    Args:
        mean (float or tuple): Mean for each channel used during normalization.
        std (float or tuple): Standard deviation for each channel used during normalization.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        """
        Args:
            img: A normalized image (as a numpy array) to be denormalized.
        
        Returns:
            The denormalized image as a numpy array.
        """
        # Denormalize the image
        denormalized_img = (img * self.std) + self.mean
        return denormalized_img



class NormalizeInference(object):
    """
    Normalize an image for inference using mean and standard deviation.

    This is adapted for inference where only one image is processed at a time,
    unlike during training where input-target pairs might be used.

    Args:
        mean (float or tuple): Mean for each channel.
        std (float or tuple): Standard deviation for each channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray or torch.Tensor): Image to be normalized.

        Returns:
            numpy.ndarray or torch.Tensor: Normalized image.
        """
        # Normalize the image
        normalized_img = (img - self.mean) / self.std
        return normalized_img
    

class MinMaxNormalize(object):
    """
    Normalize images to the 0-1 range using global minimum and maximum values provided at initialization.
    """

    def __init__(self, global_min, global_max):
        """
        Initializes the normalizer with global minimum and maximum values.

        Parameters:
        - global_min (float): The global minimum value used for normalization.
        - global_max (float): The global maximum value used for normalization.
        """
        self.global_min = global_min
        self.global_max = global_max

    def __call__(self, data):
        """
        Normalize input and target images to the 0-1 range using the global min and max.

        Args:
            data (tuple): Containing input and target images to be normalized.

        Returns:
            Tuple: Normalized input and target images.
        """
        input_img, target_img = data

        # Normalize input image
        input_normalized = (input_img - self.global_min) / (self.global_max - self.global_min)
        input_normalized = np.clip(input_normalized, 0, 1)  # Ensure within [0, 1] range

        # Normalize target image
        target_normalized = (target_img - self.global_min) / (self.global_max - self.global_min)
        target_normalized = np.clip(target_normalized, 0, 1)  # Ensure within [0, 1] range

        return input_normalized.astype(np.float32), target_normalized.astype(np.float32)



class MinMaxNormalizeInference(object):
    """
    Normalize an image to the 0-1 range using global minimum and maximum values provided at initialization.
    This is adapted for inference where only one image is processed at a time.
    
    Args:
        global_min (float): The global minimum value used for normalization.
        global_max (float): The global maximum value used for normalization.
    """

    def __init__(self, global_min, global_max):
        self.global_min = global_min
        self.global_max = global_max

    def __call__(self, img):
        """
        Normalize a single image to the 0-1 range using the global min and max.

        Args:
            img (numpy.ndarray): Image to be normalized.

        Returns:
            numpy.ndarray: Normalized image.
        """
        # Normalize the image
        normalized_img = (img - self.global_min) / (self.global_max - self.global_min)
        normalized_img = np.clip(normalized_img, 0, 1)  # Ensure within [0, 1] range
        
        return normalized_img.astype(np.float32)
    

class MinMaxNormalizeVideo(object):
    """
    Normalize images to the 0-1 range using global minimum and maximum values provided at initialization.
    This version is adapted to work with a stack of grayscale slices as input and a single grayscale slice as target,
    where each slice includes a channel dimension.
    """

    def __init__(self, global_min, global_max):
        """
        Initializes the normalizer with global minimum and maximum values.

        Parameters:
        - global_min (float): The global minimum value used for normalization.
        - global_max (float): The global maximum value used for normalization.
        """
        self.global_min = global_min
        self.global_max = global_max

    def __call__(self, data):
        """
        Apply normalization to a stack of input slices and a single target slice.

        Parameters:
        - data (tuple): A tuple containing the input stack and target slice.

        Returns:
        - Tuple: Normalized input stack and target slice.
        """
        input_stack, target_slice = data
        # Normalize each slice in the input stack
        normalized_input_stack = (input_stack - self.global_min) / (self.global_max - self.global_min)
        normalized_input_stack = np.clip(normalized_input_stack, 0, 1)
        
        # Normalize the target slice
        normalized_target_slice = (target_slice - self.global_min) / (self.global_max - self.global_min)
        normalized_target_slice = np.clip(normalized_target_slice, 0, 1)

        return normalized_input_stack, normalized_target_slice


    



class RandomSubsampleBatch(object):
    """
    Subsamples batches of images by applying a 2x2 sliding window across each image in the batch
    and randomly selecting one of the 4 pixels within each window. This process is applied
    independently to batches of input and target images, resulting in subsampled images
    with half the size of the originals.
    """

    def __call__(self, batch_data):
        """
        Subsample batches of input and target images.

        Args:
        - batch_data: a tuple containing two numpy arrays (input_patches, target_patches),
                      each with the shape (batch_size, height, width, channels).

        Returns:
        - A tuple containing subsampled batches of input and target images as NumPy arrays.
        """
        input_patches, target_patches = batch_data

        def subsample_batch(batch):
            subsampled_batch = []
            for img in batch:
                # Determine the size of the subsampled image
                new_height = img.shape[0] // 2
                new_width = img.shape[1] // 2

                # Initialize the subsampled image
                subsampled = np.zeros((new_height, new_width, img.shape[2]), dtype=img.dtype)

                # Apply 2x2 sliding window and randomly select one pixel from each window
                for i in range(new_height):
                    for j in range(new_width):
                        window = img[i*2:(i+1)*2, j*2:(j+1)*2, :]
                        random_row = np.random.randint(0, 2)
                        random_col = np.random.randint(0, 2)
                        subsampled[i, j, :] = window[random_row, random_col, :]
                
                subsampled_batch.append(subsampled)

            return np.array(subsampled_batch)

        # Subsample both input and target batches
        subsampled_input_patches = subsample_batch(input_patches)
        subsampled_target_patches = subsample_batch(target_patches)

        return subsampled_input_patches, subsampled_target_patches


class RandomSubsample(object):
    """
    Subsamples an image by applying a 2x2 sliding window across the image
    and randomly selecting one of the 4 pixels within each window. This process is applied
    independently to both an input and a target image, resulting in subsampled images
    with half the size of the originals.
    """

    def __call__(self, data):
        """
        Subsample an input and a target image.

        Args:
        - data: a tuple containing two numpy arrays (input_image, target_image),
                each with the shape (height, width, channels).

        Returns:
        - A tuple containing subsampled input and target images as NumPy arrays.
        """
        input_image, target_image = data

        def subsample_image(img):
            # Determine the size of the subsampled image
            new_height = img.shape[0] // 2
            new_width = img.shape[1] // 2

            # Initialize the subsampled image
            subsampled = np.zeros((new_height, new_width, img.shape[2]), dtype=img.dtype)

            # Apply 2x2 sliding window and randomly select one pixel from each window
            for i in range(new_height):
                for j in range(new_width):
                    window = img[i*2:(i+1)*2, j*2:(j+1)*2, :]
                    random_row = np.random.randint(0, 2)
                    random_col = np.random.randint(0, 2)
                    subsampled[i, j, :] = window[random_row, random_col, :]
            
            return subsampled

        # Subsample both the input and the target images
        subsampled_input = subsample_image(input_image)
        subsampled_target = subsample_image(target_image)

        return subsampled_input, subsampled_target
    


