import itertools
import re
from pathlib import Path

import SimpleITK as sitk
import numpy as np
import tensorflow as tf
import voxelmorph as vxm
# this version of voxelmorph is not up-to-dev branch
tf.div_no_nan = tf.math.divide_no_nan


class ImageLoader:
    
    output_image_type = np.float16
    output_segment_type = np.uint8
    lssif = sitk.LabelShapeStatisticsImageFilter()

    @staticmethod
    def load_sample_nrrd(image_filepath, segment_filepath, spacing, size):
    
        #
        # Initial loading
        #
    
        # load images
        image = sitk.ReadImage(str(image_filepath), imageIO="NrrdImageIO")
        segment = sitk.ReadImage(str(segment_filepath), imageIO="NrrdImageIO")
    
        return ImageLoader.prepare_sample(image, segment, spacing, size)[:2]

    @staticmethod
    def prepare_sample(image, segment, spacing, size):
        image = ImageLoader.rasify_and_resample(image, spacing)
        segment = ImageLoader.rasify_and_resample(segment, spacing, is_segment=True)

        crop_size = np.array(size)
        safe_crop_range, final_crop_range = ImageLoader.get_cropping_ranges(segment, crop_size)
    
        np_image_final = ImageLoader.safe_crop(image, crop_size, safe_crop_range, final_crop_range)
        np_segment_final = ImageLoader.safe_crop(segment, crop_size, safe_crop_range, final_crop_range, is_segment=True)
    
        return np_image_final, np_segment_final, safe_crop_range, final_crop_range

    @staticmethod
    def rasify_and_resample(image, spacing, is_segment=False):
        # set RAS coordinates and rescale intensities
        image_ras = ImageLoader.RASify_image(image, is_segment)
    
        # resample images
        # https://simpleitk.readthedocs.io/en/v1.2.4/Documentation/docs/source/fundamentalConcepts.html
        image_resampled = ImageLoader.resample_image(image_ras, new_spacing=spacing, is_segment=is_segment)
    
        return image_resampled

    @staticmethod
    def get_cropping_ranges(segment, crop_size):
    
        segment_size = np.array(segment.GetSize())
    
        # crop the volume
        seg_origin_resampled, seg_extent_resampled = ImageLoader.get_segment_bb(segment)
        seg_center_resampled = np.array(seg_origin_resampled) + np.array(seg_extent_resampled) // 2
    
        seg_center_resampled = segment_size - seg_center_resampled
        crop_origin = seg_center_resampled - crop_size // 2
        crop_ending = seg_center_resampled + crop_size // 2
    
        # avoid going outside the image dimensions
        safe_crop_origin = np.clip(crop_origin, [0, 0, 0], segment_size)  # np_image_resampled === np_segment_resampled
        safe_crop_ending = np.clip(crop_ending, [0, 0, 0], segment_size)
        safe_crop_range = (
            slice(safe_crop_origin[0], safe_crop_ending[0]),
            slice(safe_crop_origin[1], safe_crop_ending[1]),
            slice(safe_crop_origin[2], safe_crop_ending[2])
        )
        # since the crop_size could be bigger than the actual prostate size, we could need to add black slices to the image
        margin_origin = safe_crop_origin - crop_origin
        final_origin = margin_origin * (margin_origin != 0)
        margin_ending = safe_crop_ending - crop_ending
        final_ending = crop_size + margin_ending * (margin_ending != 0)
        final_crop_range = (
            slice(final_origin[0], final_ending[0]),
            slice(final_origin[1], final_ending[1]),
            slice(final_origin[2], final_ending[2])
        )
        return safe_crop_range, final_crop_range

    @staticmethod
    def safe_crop(image, crop_size, safe_crop_range, final_range, is_segment=False):
    
        output_type = ImageLoader.output_image_type if not is_segment else ImageLoader.output_segment_type
    
        # numpy resampled images
        np_image = ImageLoader.sitk_to_numpy(image, normalize=not is_segment).astype(output_type)
    
        # create the images of requested size and
        np_image_final = np.zeros(shape=crop_size, dtype=output_type)
        # sitk internally uses (z,y,x) coordinates, while numpy uses (x,y,z); thus, use [::-1]
        np_image_final[final_range[::-1]] = np_image[safe_crop_range[::-1]]
    
        return np_image_final

    @staticmethod
    def RASify_image(image, is_segment=False):
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
        ImageLoader.lssif.Execute(segment)
        bb = ImageLoader.lssif.GetBoundingBox(label=1)  # prostate label has value 1
        origin, extent = bb[:3], bb[3:]
        return origin, extent
    
    @staticmethod
    def sitk_to_numpy(image, normalize=True):
        # might want to read  http://simpleitk.org/SimpleITK-Notebooks/01_Image_Basics.html
        np_image = sitk.GetArrayFromImage(image)
        # this changes (x,y,z) to (z,y,x), which requires a  .transpose() , but since we applied a previous
        # change to RAS we don't need it here and we simply change the orientation flipping the axes
        #np_image = np.flip(np_image, axis=(0, 1, 2))
        # WTF does this mean?
        np_image = np_image.astype(float) / np.max(np_image) if normalize else np_image  # default minimum is 0
        return np_image

    @staticmethod
    def load_target(image_filepath, segment_filepath, spacing, size):

        output_target_type = np.uint8
        output_segment_type = np.uint8

        #
        # Initial loading
        #

        # load images
        target = sitk.ReadImage(str(image_filepath), imageIO="NrrdImageIO")
        segment = sitk.ReadImage(str(segment_filepath), imageIO="NrrdImageIO")

        # set RAS coordinates and rescale intensities
        target = ImageLoader.RASify_image(target, is_segment=True)
        segment = ImageLoader.RASify_image(segment, is_segment=True)

        # resample images
        # https://simpleitk.readthedocs.io/en/v1.2.4/Documentation/docs/source/fundamentalConcepts.html
        target_resampled = ImageLoader.resample_image(target, new_spacing=spacing, is_segment=True)
        segment_resampled = ImageLoader.resample_image(segment, new_spacing=spacing, is_segment=True)

        # numpy resampled images
        np_target_resampled = ImageLoader.sitk_to_numpy(target_resampled, normalize=False).astype(output_target_type)
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

        safe_crop_origin = np.clip(crop_origin, [0, 0, 0], np_target_resampled.shape)
        safe_crop_ending = np.clip(crop_ending, [0, 0, 0], np_target_resampled.shape)
        np_target_cropped = np_target_resampled[safe_crop_origin[0]:safe_crop_ending[0], safe_crop_origin[1]:safe_crop_ending[1],safe_crop_origin[2]:safe_crop_ending[2]]
        np_segment_cropped = np_segment_resampled[safe_crop_origin[0]:safe_crop_ending[0], safe_crop_origin[1]:safe_crop_ending[1],safe_crop_origin[2]:safe_crop_ending[2]]

        # since the crop_size could be bigger than the actual prostate size, we could need to add black slices to the image
        final_origin = - (crop_origin - safe_crop_origin) * ((crop_origin - safe_crop_origin) != 0)
        final_ending = crop_size - (crop_ending - safe_crop_ending) * ((crop_ending - safe_crop_ending) != 0)

        np_target_final = np.zeros(shape=crop_size, dtype=output_target_type)
        np_target_final[final_origin[0]:final_ending[0], final_origin[1]:final_ending[1], final_origin[2]:final_ending[2]] = np_target_cropped

        return np_target_final, target


class DatasetMaker:

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
            mr_image_cropped, mr_prostate_cropped = ImageLoader.load_sample_nrrd(mr_image_filename, mr_prostate_filename, self.resampling_spacing, self.dim)
            us_image_cropped, us_prostate_cropped = ImageLoader.load_sample_nrrd(us_image_filename, us_prostate_filename, self.resampling_spacing, self.dim)
            # save to file
            data = {
                "mr_image": mr_image_cropped,
                "mr_seg": mr_prostate_cropped,
                "us_image": us_image_cropped,
                "us_seg": us_prostate_cropped
            }
            outfile = str(output_folder / f"{patient}_{mr_data}_{us_data}.npz")
            np.savez_compressed(outfile, **data)

    @staticmethod
    def check(output_folder: Path):
        for data_path in output_folder.iterdir():
            data = np.load(str(data_path))
            try:
                _, _, _, _ = data['mr_image'], data['mr_seg'], data['us_image'], data['us_seg']
            except Exception as ex:
                print(f"File {data_path.name} not well compressed!")

    def add_target(self, path, patient, mr_data, us_data):
            mr_targets = sorted((self.folder_mr / patient / mr_data).rglob("Target*"))
            us_targets = sorted((self.folder_us / patient / us_data).rglob("Target*"))
            mr_prostate_filename = list((self.folder_mr / patient / mr_data).rglob("Prostate*"))[0]
            us_prostate_filename = list((self.folder_us / patient / us_data).rglob("Prostate*"))[0]
            npz = dict(np.load(path))
            for mr_target in mr_targets:
                mr_name = mr_target.name.split("_")[0].lower()
                mr_target_cropped, _ = ImageLoader.load_target(mr_target, mr_prostate_filename, self.resampling_spacing, self.dim)
                npz[f"mr_{mr_name}"] = mr_target_cropped
            for us_target in us_targets:
                us_name = us_target.name.split("_")[0].lower()
                us_target_cropped, _ = ImageLoader.load_target(us_target, us_prostate_filename, self.resampling_spacing, self.dim)
                npz[f"us_{us_name}"] = us_target_cropped
            np.savez_compressed(path, **npz)

    def add_targets(self, output_path):
        for f in output_path.iterdir():
            fn = f.stem.split("_")
            patient, mr_data, us_data = fn[0], fn[1], fn[2]
            self.add_target(f, patient, mr_data, us_data)
            print(f"done {patient}")

    def check_targets(self, output_path):
        from tqdm import tqdm
        for f in tqdm(list(output_path.iterdir())):
            npz = dict(np.load(f))

            re_filter_mr = re.compile('^mr_target.*$')
            re_filter_us = re.compile('^us_target.*$')

            tags = npz.keys()

            mr_tags = [s for s in tags if re.match(re_filter_mr, s)]
            us_tags = [s for s in tags if re.match(re_filter_us, s)]

            if len(mr_tags) == 0 or len(us_tags) == 0:
                print(f"---> BAD {f.stem}, saving it again")
                fn = f.stem.split("_")
                patient, mr_data, us_data = fn[0], fn[1], fn[2]
                self.add_target(f, patient, mr_data, us_data)
                print(f"done {patient}")


class Generator(tf.keras.utils.Sequence):
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
        inputs, outputs, mr_targets, us_targets = self._data_generation(patients_list, output_targets=output_targets)

        return inputs, outputs, mr_targets, us_targets, patients_list

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        if self.shuffle:
            if self.seed is not None:
                np.random.seed(self.seed)
            np.random.shuffle(self.indexes)

    def _data_generation(self, patients_list, output_targets=False):
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
            mr_image_crop, mr_prostate_crop, us_image_crop, us_prostate_crop, targets = Generator.single_load(data_path)
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
        mr_image, mr_prostate, us_image, us_prostate, _ = Generator.single_load(path)
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


def dice_coeff(image1, image2):
    return (2 * np.sum(image1 * image2) + 1) / (np.sum(image1 + image2) + 1)


def prepare_model(inshape, sim_param=1.0, lambda_param=0.05, gamma_param=0.01):

    tf.keras.backend.clear_session()

    # same as default
    enc_nf = [16, 32, 32, 32]
    dec_nf = [32, 32, 32, 32, 32, 16, 16]
    
    vxm_model = vxm.networks.VxmDenseSemiSupervisedSeg(
        inshape=inshape, nb_labels=1,
        # nb_unet_features=[enc_nf, dec_nf],
        seg_downsize=2
    )

    # assigning loss
    bin_centers = np.linspace(0, 1, 32)  # histogram bins, assume normalized images
    loss_mi = vxm.losses.NMI(bin_centers=bin_centers, vol_size=inshape).loss
    loss_smooth = vxm.losses.Grad('l2').loss
    loss_dice = vxm.losses.Dice().loss
    losses = [loss_mi, loss_smooth, loss_dice]
    loss_weights = [sim_param, lambda_param, gamma_param]

    # assigning metrics
    @tf.function
    def mi(y_true, y_pred):
        return -loss_mi(y_true, y_pred)
    @tf.function
    def dice(y_true, y_pred):
        return -loss_dice(y_true, y_pred)
    metrics = {'transformer': [mi],
               'seg_transformer': [dice]}

    vxm_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=losses, loss_weights=loss_weights,
        metrics=metrics
    )
    return vxm_model


class DatasetCreator:
    
    @staticmethod
    def split_dataset(dataset_folder, train_test_split=0.95, train_val_split=0.90, seed=None):
        
        full_data = np.array(dataset_folder.iterdir())
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(full_data)
        
        # split train/test
        idx = [int(train_test_split * full_data.shape[0])]
        train_data, test_data = np.split(full_data, idx)
    
        # split train/validation
        idx = [int(train_val_split * train_data.shape[0])]
        train_data, validation_data = np.split(train_data, idx)
    
        return train_data, validation_data, test_data

#
# EVALUATION
#


def get_index(string, shift=0):
    if string.startswith('0.0'):
        return 0
    if string == '0.5':
        return 1
    if string == '1.0':
        return 2
    if string == '2.0':
        return 3
    if string == '3.0':
        return 4
    if string == '4.0':
        return 5
    if string == '5.0':
        return 6
    return None


def init_stats(n, m):
    grid_shape = (n, m)
    stats = {
        "prostate": {
            "pre": {
                "dice": {
                    "mean": np.full(shape=grid_shape, fill_value=np.nan),
                    "std": np.full(shape=grid_shape, fill_value=np.nan)
                }
            },
            "def": {
                "dice": {
                    "mean": np.full(shape=grid_shape, fill_value=np.nan),
                    "std": np.full(shape=grid_shape, fill_value=np.nan)
                }
            },
            "final": {
                "dice": {
                    "mean": np.full(shape=grid_shape, fill_value=np.nan),
                    "std": np.full(shape=grid_shape, fill_value=np.nan)
                }
            },
        },
        "target": {
            "pre": {
                "dice": {
                    "mean": np.full(shape=grid_shape, fill_value=np.nan),
                    "std": np.full(shape=grid_shape, fill_value=np.nan)
                },
                "error": {
                    "mean": np.full(shape=grid_shape, fill_value=np.nan),
                    "std": np.full(shape=grid_shape, fill_value=np.nan)
                }
            },
            "def": {
                "valid": {
                    "dice": {
                        "mean": np.full(shape=grid_shape, fill_value=np.nan),
                        "std": np.full(shape=grid_shape, fill_value=np.nan)
                    },
                    "error": {
                        "mean": np.full(shape=grid_shape, fill_value=np.nan),
                        "std": np.full(shape=grid_shape, fill_value=np.nan)
                    }
                },
                "good": {
                    "dice": {
                        "mean": np.full(shape=grid_shape, fill_value=np.nan),
                        "std": np.full(shape=grid_shape, fill_value=np.nan)
                    },
                    "error": {
                        "mean": np.full(shape=grid_shape, fill_value=np.nan),
                        "std": np.full(shape=grid_shape, fill_value=np.nan)
                    }
                },
                "decent": {
                    "dice": {
                        "mean": np.full(shape=grid_shape, fill_value=np.nan),
                        "std": np.full(shape=grid_shape, fill_value=np.nan)
                    },
                    "error": {
                        "mean": np.full(shape=grid_shape, fill_value=np.nan),
                        "std": np.full(shape=grid_shape, fill_value=np.nan)
                    }
                },
                "bad": {
                    "dice": {
                        "mean": np.full(shape=grid_shape, fill_value=np.nan),
                        "std": np.full(shape=grid_shape, fill_value=np.nan)
                    },
                    "error": {
                        "mean": np.full(shape=grid_shape, fill_value=np.nan),
                        "std": np.full(shape=grid_shape, fill_value=np.nan)
                    }
                }
            },
            "final": {
                "valid": {
                    "ratio": np.full(shape=grid_shape, fill_value=np.nan),
                    "dice": {
                        "mean": np.full(shape=grid_shape, fill_value=np.nan),
                        "std": np.full(shape=grid_shape, fill_value=np.nan)
                    },
                    "error": {
                        "mean": np.full(shape=grid_shape, fill_value=np.nan),
                        "std": np.full(shape=grid_shape, fill_value=np.nan)
                    }
                },
                "good": {
                    "ratio": np.full(shape=grid_shape, fill_value=np.nan),
                    "dice": {
                        "mean": np.full(shape=grid_shape, fill_value=np.nan),
                        "std": np.full(shape=grid_shape, fill_value=np.nan)
                    },
                    "error": {
                        "mean": np.full(shape=grid_shape, fill_value=np.nan),
                        "std": np.full(shape=grid_shape, fill_value=np.nan)
                    }
                },
                "decent": {
                    "ratio": np.full(shape=grid_shape, fill_value=np.nan),
                    "dice": {
                        "mean": np.full(shape=grid_shape, fill_value=np.nan),
                        "std": np.full(shape=grid_shape, fill_value=np.nan)
                    },
                    "error": {
                        "mean": np.full(shape=grid_shape, fill_value=np.nan),
                        "std": np.full(shape=grid_shape, fill_value=np.nan)
                    }
                },
                "bad": {
                    "ratio": np.full(shape=grid_shape, fill_value=np.nan),
                    "dice": {
                        "mean": np.full(shape=grid_shape, fill_value=np.nan),
                        "std": np.full(shape=grid_shape, fill_value=np.nan)
                    },
                    "error": {
                        "mean": np.full(shape=grid_shape, fill_value=np.nan),
                        "std": np.full(shape=grid_shape, fill_value=np.nan)
                    }
                }
            }
        }
    }
    return stats


def store_stats(stats, results, l, g, good_dice_threshold=0.25):
    # PROSTATE: PRE AND DEF (only Dice)
    stats["prostate"]["pre"]["dice"]["mean"][l, g] = results['prostate_dice_pre'].mean()
    stats["prostate"]["pre"]["dice"]["std"][l, g] = results['prostate_dice_pre'].std()
    stats["prostate"]["def"]["dice"]["mean"][l, g] = results['prostate_dice_def'].mean()
    stats["prostate"]["def"]["dice"]["std"][l, g] = results['prostate_dice_def'].std()
    # PROSTATE: FINAL (only Dice)
    stats["prostate"]["final"]["dice"]["mean"][l, g] = results['prostate_dice'].mean()
    stats["prostate"]["final"]["dice"]["std"][l, g] = results['prostate_dice'].std()

    # TARGET: PRE AND DEF
    stats["target"]["pre"]["dice"]["mean"][l, g] = results['target_dice_pre'].mean()
    stats["target"]["pre"]["dice"]["std"][l, g] = results['target_dice_pre'].std()
    stats["target"]["pre"]["error"]["mean"][l, g] = results['target_error_pre'].mean()
    stats["target"]["pre"]["error"]["std"][l, g] = results['target_error_pre'].std()
    # TARGET: FINAL (valid)
    valid_dice_vals = results['target_dice'][np.isfinite(results['target_dice'])]
    valid_error_vals = results['target_error'][np.isfinite(results['target_error'])]
    stats["target"]["final"]["valid"]["ratio"][l, g] = len(valid_dice_vals) / len(results['target_dice'])
    stats["target"]["final"]["valid"]["dice"]["mean"][l, g] = valid_dice_vals.mean()
    stats["target"]["final"]["valid"]["dice"]["std"][l, g] = valid_dice_vals.std()
    stats["target"]["final"]["valid"]["error"]["mean"][l, g] = valid_error_vals.mean()
    stats["target"]["final"]["valid"]["error"]["std"][l, g] = valid_error_vals.std()
    valid_dice_vals = results['target_dice_def'][np.isfinite(results['target_dice'])]
    valid_error_vals = results['target_error_def'][np.isfinite(results['target_error'])]
    stats["target"]["def"]["valid"]["dice"]["mean"][l, g] = valid_dice_vals.mean()
    stats["target"]["def"]["valid"]["dice"]["std"][l, g] = valid_dice_vals.std()
    stats["target"]["def"]["valid"]["error"]["mean"][l, g] = valid_error_vals.mean()
    stats["target"]["def"]["valid"]["error"]["std"][l, g] = valid_error_vals.std()
    # TARGET: FINAL (good)
    good_dice_vals = results['target_dice'][results['target_dice'] > good_dice_threshold]
    good_error_vals = results['target_error'][results['target_dice'] > good_dice_threshold]
    stats["target"]["final"]["good"]["ratio"][l, g] = len(good_dice_vals) / len(results['target_dice'])
    stats["target"]["final"]["good"]["dice"]["mean"][l, g] = good_dice_vals.mean()
    stats["target"]["final"]["good"]["dice"]["std"][l, g] = good_dice_vals.std()
    stats["target"]["final"]["good"]["error"]["mean"][l, g] = good_error_vals.mean()
    stats["target"]["final"]["good"]["error"]["std"][l, g] = good_error_vals.std()
    good_dice_vals = results['target_dice_def'][results['target_dice'] > good_dice_threshold]
    good_error_vals = results['target_error_def'][results['target_dice'] > good_dice_threshold]
    stats["target"]["def"]["good"]["dice"]["mean"][l, g] = good_dice_vals.mean()
    stats["target"]["def"]["good"]["dice"]["std"][l, g] = good_dice_vals.std()
    stats["target"]["def"]["good"]["error"]["mean"][l, g] = good_error_vals.mean()
    stats["target"]["def"]["good"]["error"]["std"][l, g] = good_error_vals.std()
    # TARGET: FINAL (decent)
    decent_dice_vals = results['target_dice'][results['target_dice'] > 0]
    decent_error_vals = results['target_error'][results['target_dice'] > 0]
    stats["target"]["final"]["decent"]["ratio"][l, g] = len(decent_dice_vals) / len(results['target_dice'])
    stats["target"]["final"]["decent"]["dice"]["mean"][l, g] = decent_dice_vals.mean()
    stats["target"]["final"]["decent"]["dice"]["std"][l, g] = decent_dice_vals.std()
    stats["target"]["final"]["decent"]["error"]["mean"][l, g] = decent_error_vals.mean()
    stats["target"]["final"]["decent"]["error"]["std"][l, g] = decent_error_vals.std()
    decent_dice_vals = results['target_dice_def'][results['target_dice'] > 0]
    decent_error_vals = results['target_error_def'][results['target_dice'] > 0]
    stats["target"]["def"]["decent"]["dice"]["mean"][l, g] = decent_dice_vals.mean()
    stats["target"]["def"]["decent"]["dice"]["std"][l, g] = decent_dice_vals.std()
    stats["target"]["def"]["decent"]["error"]["mean"][l, g] = decent_error_vals.mean()
    stats["target"]["def"]["decent"]["error"]["std"][l, g] = decent_error_vals.std()
    # TARGET: FINAL (bad)
    bad_dice_vals = results['target_dice'][results['target_dice'] == 0]
    bad_error_vals = results['target_error'][results['target_dice'] == 0]
    stats["target"]["final"]["bad"]["ratio"][l, g] = len(bad_dice_vals) / len(results['target_dice'])
    stats["target"]["final"]["bad"]["dice"]["mean"][l, g] = bad_dice_vals.mean()
    stats["target"]["final"]["bad"]["dice"]["std"][l, g] = bad_dice_vals.std()
    stats["target"]["final"]["bad"]["error"]["mean"][l, g] = bad_error_vals.mean()
    stats["target"]["final"]["bad"]["error"]["std"][l, g] = bad_error_vals.std()
    bad_dice_vals = results['target_dice_def'][results['target_dice'] == 0]
    bad_error_vals = results['target_error_def'][results['target_dice'] == 0]
    stats["target"]["def"]["bad"]["dice"]["mean"][l, g] = bad_dice_vals.mean()
    stats["target"]["def"]["bad"]["dice"]["std"][l, g] = bad_dice_vals.std()
    stats["target"]["def"]["bad"]["error"]["mean"][l, g] = bad_error_vals.mean()
    stats["target"]["def"]["bad"]["error"]["std"][l, g] = bad_error_vals.std()

    return stats


def print_stats(stats, l, g):
    print(f"prostate Dice PRE:   {stats['prostate']['pre']['dice']['mean'][l, g] * 100:.2f}±{stats['prostate']['pre']['dice']['std'][l, g] * 100:.2f}%")
    print(f"prostate Dice INTRA: {stats['prostate']['def']['dice']['mean'][l, g] * 100:.2f}±{stats['prostate']['def']['dice']['std'][l, g] * 100:.2f}%")
    print(f"prostate Dice POST:  {stats['prostate']['final']['dice']['mean'][l, g] * 100:.2f}±{stats['prostate']['final']['dice']['std'][l, g] * 100:.2f}%")
    print()
    print()
    print(f"target Dice PRE:   {stats['target']['pre']['dice']['mean'][l, g] * 100:.2f}±{stats['target']['pre']['dice']['std'][l, g] * 100:.2f}%")
    print(f"TRE PRE:           {stats['target']['pre']['error']['mean'][l, g]:.2f}±{stats['target']['pre']['error']['std'][l, g]:.2f}mm")
    print()
    for metric in ["valid", "good", "decent", "bad"]:
        print(f"# {metric} target Dice: {stats['target']['final'][metric]['ratio'][l, g] * 100:.2f}%")
        print(f"target Dice INTRA:      {stats['target']['def'][metric]['dice']['mean'][l, g] * 100:.2f}±{stats['target']['def'][metric]['dice']['std'][l, g] * 100:.2f}%")
        print(f"TRE INTRA:              {stats['target']['def'][metric]['error']['mean'][l, g]:.2f}±{stats['target']['def'][metric]['error']['std'][l, g]:.2f}mm")
        print(f"{metric} Dice:          {stats['target']['final'][metric]['dice']['mean'][l, g] * 100:.2f}±{stats['target']['final'][metric]['dice']['std'][l, g] * 100:.2f}%")
        print(f"{metric} TRE:           {stats['target']['final'][metric]['error']['mean'][l, g]:.2f}±{stats['target']['final'][metric]['error']['std'][l, g]:.2f}mm")
        print()
