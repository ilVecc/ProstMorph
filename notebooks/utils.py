import itertools
from pathlib import Path

import SimpleITK as sitk
import numpy as np


class ImageLoader:
    
    @staticmethod
    def sitk_to_numpy(image, normalize=True):
        # might want to read  http://simpleitk.org/SimpleITK-Notebooks/01_Image_Basics.html
        # this changes (x,y,z) to (z,y,x), which requires a  .transpose() , but since we applied a previous change to RAS we don't need it here and we simply
        # change the orientation flipping the axes
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
