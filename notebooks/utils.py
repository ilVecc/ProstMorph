import itertools
import re
from pathlib import Path

import SimpleITK as sitk
import numpy as np
import tensorflow as tf


class ImageLoader:
    
    @staticmethod
    def sitk_to_numpy(image, normalize=True):
        # might want to read  http://simpleitk.org/SimpleITK-Notebooks/01_Image_Basics.html
        # this changes (x,y,z) to (z,y,x), which requires a  .transpose() , but since we applied a previous
        # change to RAS we don't need it here and we simply change the orientation flipping the axes
        np_image = sitk.GetArrayFromImage(image)
        np_image = np.flip(np_image, axis=(0, 1, 2))
        np_image = np_image.astype(float) / np_image.max() if normalize else np_image  # default minimum is 0
        return np_image
    
    @staticmethod
    def rasify_image(image, is_segment=False):
        doif = sitk.DICOMOrientImageFilter()
        doif.SetDesiredCoordinateOrientation('RAS')
        image = doif.Execute(image)
        image = sitk.RescaleIntensity(image, 0, 255) if not is_segment else image
        return image
    
    @staticmethod
    def resample_image(image, new_size=None, new_spacing=None, is_segment=False):
        if new_size is None and new_spacing is None:
            raise Exception("At least one between size and spacing must be specified")
        if new_size is not None and new_spacing is not None:
            raise Exception("Either size or spacing must be specified")
        resample = sitk.ResampleImageFilter()
        resample.SetInterpolator(sitk.sitkLinear if not is_segment else sitk.sitkNearestNeighbor)
        resample.SetOutputDirection(image.GetDirection())
        resample.SetOutputOrigin(image.GetOrigin())
        
        orig_size = np.array(image.GetSize(), dtype=int)
        orig_spacing = np.array(image.GetSpacing())
        if new_size is not None:
            new_spacing = (orig_spacing * orig_size / new_size).tolist()
        else:
            new_size = (orig_size * orig_spacing / new_spacing).astype(int).tolist()
        resample.SetSize(new_size)
        resample.SetOutputSpacing(new_spacing)
        
        new_image = resample.Execute(image)
        return new_image
    
    @staticmethod
    def get_segment_bb(segment):
        lssif = sitk.LabelShapeStatisticsImageFilter()
        lssif.Execute(segment)
        bb = lssif.GetBoundingBox(1)  # prostate label has value 1
        origin, extent = bb[:3], bb[3:]
        return origin, extent
    
    @staticmethod
    def load_image(image_filepath, segment_filepath, spacing, size):
        
        output_image_type = np.float16
        output_segment_type = np.uint8
        
        #
        # Initial loading
        #
        
        # load images
        image = sitk.ReadImage(str(image_filepath), imageIO="NrrdImageIO")
        segment = sitk.ReadImage(str(segment_filepath), imageIO="NrrdImageIO")
        
        # set RAS coordinates and rescale intensities
        image = ImageLoader.rasify_image(image)
        segment = ImageLoader.rasify_image(segment, is_segment=True)
        
        # resample images
        # https://simpleitk.readthedocs.io/en/v1.2.4/Documentation/docs/source/fundamentalConcepts.html
        image_resampled = ImageLoader.resample_image(image, new_spacing=spacing)
        segment_resampled = ImageLoader.resample_image(segment, new_spacing=spacing, is_segment=True)
        
        # numpy resampled images
        np_image_resampled = ImageLoader.sitk_to_numpy(image_resampled).astype(output_image_type)
        np_segment_resampled = ImageLoader.sitk_to_numpy(segment_resampled, normalize=False).astype(output_segment_type)
        
        #
        # Final cropping
        #
        
        # crop the volume
        seg_origin_resampled, seg_extent_resampled = ImageLoader.get_segment_bb(segment_resampled)
        seg_center_resampled = np.array(seg_origin_resampled) + np.array(seg_extent_resampled) // 2
        crop_size = np.array(size)
        
        # since in the  sitk_to_numpy()  we don't use the transpose, the  crop_*  variables must refer to the (z,y,x) coordinates; thus, use [::-1]
        # also, we must account for the  .flip() , so we also reverse the coordinates
        seg_center_resampled = np.array(np_segment_resampled.shape) - seg_center_resampled[::-1]
        crop_origin = seg_center_resampled - crop_size[::-1] // 2
        crop_ending = seg_center_resampled + crop_size[::-1] // 2
        
        safe_crop_origin = np.clip(crop_origin, [0, 0, 0], np_image_resampled.shape)
        safe_crop_ending = np.clip(crop_ending, [0, 0, 0], np_image_resampled.shape)
        np_image_cropped = np_image_resampled[safe_crop_origin[0]:safe_crop_ending[0], safe_crop_origin[1]:safe_crop_ending[1], safe_crop_origin[2]:safe_crop_ending[2]]
        np_segment_cropped = np_segment_resampled[safe_crop_origin[0]:safe_crop_ending[0], safe_crop_origin[1]:safe_crop_ending[1], safe_crop_origin[2]:safe_crop_ending[2]]
        
        # since the crop_size could be bigger than the actual prostate size, we could need to add black slices to the image
        final_origin = - (crop_origin - safe_crop_origin) * ((crop_origin - safe_crop_origin) != 0)
        final_ending = crop_size - (crop_ending - safe_crop_ending) * ((crop_ending - safe_crop_ending) != 0)
        
        np_image_final = np.zeros(shape=crop_size, dtype=output_image_type)
        np_segment_final = np.zeros(shape=crop_size, dtype=output_segment_type)
        np_image_final[final_origin[0]:final_ending[0], final_origin[1]:final_ending[1], final_origin[2]:final_ending[2]] = np_image_cropped
        np_segment_final[final_origin[0]:final_ending[0], final_origin[1]:final_ending[1], final_origin[2]:final_ending[2]] = np_segment_cropped
        
        return np_image_final, np_segment_final, image, segment


class DatasetCreator:
    
    def __init__(self, dataset_path: Path, dim: tuple, resampling_spacing: tuple):
        self.folder_base = dataset_path
        self.folder_mr = self.folder_base / "mri"
        self.folder_us = self.folder_base / "us"
        self.resampling_spacing = resampling_spacing
        self.dim = dim
        
        patients_us_list = [f.name for f in self.folder_us.iterdir()]
        patients_list = [f.name for f in self.folder_mr.iterdir() if f.name in patients_us_list]
        self.patients_options = [
            (p, mr, us)
            for p in patients_list
            for mr, us in list(itertools.product([f.name for f in (self.folder_mr / p).iterdir()], [f.name for f in (self.folder_us / p).iterdir()]))
        ]
    
    def run(self, output_folder: Path):
        output_folder.mkdir(exist_ok=True)
        for i, (patient, mr_data, us_data) in enumerate(self.patients_options):
            mr_image_filename = list((self.folder_mr / patient / mr_data).rglob("MRI*"))[0]
            us_image_filename = list((self.folder_us / patient / us_data).rglob("US*"))[0]
            mr_prostate_filename = list((self.folder_mr / patient / mr_data).rglob("Prostate*"))[0]
            us_prostate_filename = list((self.folder_us / patient / us_data).rglob("Prostate*"))[0]
            mr_image_cropped, mr_prostate_cropped, _, _ = ImageLoader.load_image(mr_image_filename, mr_prostate_filename, self.resampling_spacing, self.dim)
            us_image_cropped, us_prostate_cropped, _, _ = ImageLoader.load_image(us_image_filename, us_prostate_filename, self.resampling_spacing, self.dim)
            # save to file
            outfile = str(output_folder / f"{patient}_{mr_data}_{us_data}.npz")
            np.savez_compressed(outfile, mr_image=mr_image_cropped, mr_seg=mr_prostate_cropped, us_image=us_image_cropped, us_seg=us_prostate_cropped)
    
    @staticmethod
    def check(output_folder: Path):
        for data_path in output_folder.iterdir():
            data = np.load(str(data_path))
            try:
                _, _, _, _ = data['mr_image'], data['mr_seg'], data['us_image'], data['us_seg']
            except Exception as ex:
                print(f"File {data_path.name} not well compressed!")


class SmartDataGenerator(tf.keras.utils.Sequence):
    """Generates data for Keras"""
    def __init__(self, data_paths, dim, batch_size=32, shuffle=True, seed=None):
        """Initialization"""
        self.dim = dim
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

        self.patients_cases = data_paths

        self.indexes = np.arange(len(self.patients_cases))
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.patients_cases) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        inputs, outputs, _, _, _ = self(index, output_targets=False)
        return inputs, outputs

    def __call__(self, index, output_targets=True):
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        patients_list = [self.patients_cases[k] for k in indexes]

        # Generate data
        inputs, outputs, mr_targets, us_targets = self.__data_generation(patients_list, output_targets=output_targets)

        return inputs, outputs, mr_targets, us_targets, patients_list

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        if self.shuffle:
            if self.seed is not None:
                np.random.seed(self.seed)
            np.random.shuffle(self.indexes)

    def __data_generation(self, patients_list, output_targets=False):
        """Generates data containing batch_size samples"""
        # X : (n_samples, *dim, n_channels)
        moving_images = np.zeros(shape=(self.batch_size, *self.dim, 1))
        fixed_images = np.zeros(shape=(self.batch_size, *self.dim, 1))
        moving_images_seg = np.zeros(shape=(self.batch_size, *[d // 2 for d in self.dim], 1))
        fixed_images_seg = np.zeros(shape=(self.batch_size, *[d // 2 for d in self.dim], 1))
        zero_phi = np.zeros(shape=(self.batch_size, *self.dim, len(self.dim)))
        mr_targets = [None] * self.batch_size
        us_targets = [None] * self.batch_size

        # Generate data
        re_filter_mr = re.compile('^mr_target.*$')
        re_filter_us = re.compile('^us_target.*$')
        for i, data_path in enumerate(patients_list):
            mr_image_crop, mr_prostate_crop, us_image_crop, us_prostate_crop, targets = \
                SmartDataGenerator.single_load(data_path)
            # images need to be of the size [batch_size, H, W, D, 1]
            moving_images[i, ..., 0] = mr_image_crop
            fixed_images[i, ..., 0] = us_image_crop
            moving_images_seg[i, ..., 0] = mr_prostate_crop[::2, ::2, ::2]
            fixed_images_seg[i, ..., 0] = us_prostate_crop[::2, ::2, ::2]
            # add targets
            if output_targets:
                target_names = targets.keys()
                mr_target_names = sorted([s for s in target_names if re_filter_mr.match(s)])
                tar = [targets[mr_target_name] for mr_target_name in mr_target_names]
                if len(tar) > 0:
                    mr_targets[i] = np.concatenate([t[..., np.newaxis] for t in tar], axis=3)
                else:
                    mr_targets[i] = None
                us_target_names = sorted([s for s in target_names if re_filter_us.match(s)])
                tar = [targets[us_target_name] for us_target_name in us_target_names]
                if len(tar) > 0:
                    us_targets[i] = np.concatenate([t[..., np.newaxis] for t in tar], axis=3)
                else:
                    us_targets[i] = None

        inputs = [moving_images, fixed_images, moving_images_seg]

        # prepare outputs (the 'true' moved image):
        # of course, we don't have this, but we know we want to compare
        # the resulting moved image with the fixed image.
        # we also wish to penalize the deformation field.
        outputs = [fixed_images, zero_phi, fixed_images_seg]

        return inputs, outputs, mr_targets, us_targets

    @staticmethod
    def single_load(path: str):
        data = dict(np.load(path))
        mr_image, mr_prostate = data['mr_image'], data['mr_seg']
        us_image, us_prostate = data['us_image'], data['us_seg']
        del data['mr_image'], data['mr_seg'], data['us_image'], data['us_seg']
        return mr_image, mr_prostate, us_image, us_prostate, data

    @staticmethod
    def single_input(path):
        mr_image, mr_prostate, us_image, us_prostate, _ = SmartDataGenerator.single_load(path)
        return [mr_image, us_image, mr_prostate]


class MIND_SSC:
    def pdist_squared(self, x):
        xx = (x**2).sum(dim=1).unsqueeze(2)
        yy = xx.permute(0, 2, 1)
        dist = xx + yy - 2.0 * tf.matmul(x.permute(0, 2, 1), x)
        dist[dist != dist] = 0
        dist = tf.clip_by_value(dist, 0.0, np.inf)
        return dist

    def MINDSSC(self, img, radius=2, dilation=2):
        # see http://mpheinrich.de/pub/miccai2013_943_mheinrich.pdf for details on the MIND-SSC descriptor

        # kernel size
        kernel_size = radius * 2 + 1

        # define start and end locations for self-similarity pattern
        six_neighbourhood = tf.Tensor([[0, 1, 1],
                                       [1, 1, 0],
                                       [1, 0, 1],
                                       [1, 1, 2],
                                       [2, 1, 1],
                                       [1, 2, 1]], dtype=tf.int64)

        # squared distances
        dist = tf.squeeze(self.pdist_squared(tf.expand_dims(tf.transpose(six_neighbourhood), axis=0)), axis=0)

        # define comparison mask
        x, y = tf.meshgrid(tf.range(6), tf.range(6))
        mask = (x > y) & (dist == 2)

        # build kernel
        idx_shift1 = tf.expand_dims(six_neighbourhood, 1).repeat(1, 6, 1).reshape(-1, 3)[mask, :]
        mshift1 = tf.zeros(shape=(12, 1, 3, 3, 3))
        mshift1.view(-1)[tf.range(12) * 27 + idx_shift1[:, 0] * 9 + idx_shift1[:, 1] * 3 + idx_shift1[:, 2]] = 1

        idx_shift2 = tf.expand_dims(six_neighbourhood, 0).repeat(6, 1, 1).reshape(-1, 3)[mask, :]
        mshift2 = tf.zeros(shape=(12, 1, 3, 3, 3))
        mshift2.view(-1)[tf.range(12) * 27 + idx_shift2[:, 0] * 9 + idx_shift2[:, 1] * 3 + idx_shift2[:, 2]] = 1

        rpad1 = tf.pad(img, paddings=dilation, mode='SYMMETRIC')

        # compute patch-ssd
        ssd = tf.nn.avg_pool3d(
            tf.pad(
                (tf.nn.conv3d(rpad1, filters=mshift1, dilation=dilation) - tf.nn.conv3d(rpad1, filters=mshift2, dilation=dilation)) ** 2,
                paddings=radius,
                mode='SYMMETRIC'),
            kernel_size,
            stride=1)

        # MIND equation
        mind = ssd - tf.reduce_min(ssd, axis=1, keepdims=True)[0]
        mind_var = tf.reduce_mean(mind, axis=1, keepdims=True)
        mind_var = tf.clip_by_value(mind_var, mind_var.mean()*0.001, mind_var.mean()*1000)
        mind /= mind_var
        mind = tf.exp(-mind)

        # permute to have same ordering as C++ code
        mind = mind[:, tf.Tensor([6, 8, 1, 11, 2, 10, 0, 7, 9, 4, 5, 3], dtype=tf.int64), :, :, :]

        return mind

    def mind_loss(self, x, y):
        return tf.reduce_mean((self.MINDSSC(x) - self.MINDSSC(y)) ** 2)


def dice(image1, image2):
    return 2 * np.bitwise_and(image1 != 0, image2 != 0).sum() / (image1.sum() + image2.sum())

