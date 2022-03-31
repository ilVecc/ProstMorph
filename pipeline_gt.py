#!/usr/bin/env python-real

import slicer
import json
from pathlib import Path


save_df = False
compute_coeffs = True


def process(path_mr_image, path_mr_seg, path_us_image, path_us_seg):

    slicer.mrmlScene.Clear()

    node_mr_image = slicer.util.loadVolume(path_mr_image)
    node_mr_seg = slicer.util.loadSegmentation(path_mr_seg)
    node_us_image = slicer.util.loadVolume(path_us_image)
    node_us_seg = slicer.util.loadSegmentation(path_us_seg)

    widget = slicer.modules.prostatemriuscontourpropagation.widgetRepresentation().self()
    widget.mrVolumeNodeCombobox.setCurrentNode(node_mr_image)
    widget.mrSegmentationNodeCombobox.setCurrentNode(node_mr_seg)
    widget.usVolumeNodeCombobox.setCurrentNode(node_us_image)
    widget.usSegmentationNodeCombobox.setCurrentNode(node_us_seg)

    widget.performRegistrationButton.click()
    widget.noRegistrationRadioButton.click()
    node_pre_trans = slicer.util.getNode("PreAlign*")
    node_pre_trans.SetName("PreAlign Transform")
    node_def_trans = slicer.util.getNode("Deformable*")
    node_def_trans.SetName("Deformation Transform")
    
    # save transforms as displacement fields
    if save_df:
        node_pre_field = slicer.modules.transforms.logic().ConvertToGridTransform(node_pre_trans, node_mr_image)
        node_pre_field.SetName("PreAlign DF")
        node_pre_trans = node_pre_field
        node_def_field = slicer.modules.transforms.logic().ConvertToGridTransform(node_def_trans, node_mr_image)
        node_def_field.SetName("Deformation DF")
        node_def_trans = node_def_field

    # calculate DICE and Hausenhoff coefficient
    widget.logic.calculateSegmentSimilarity()
    dice_coeff = -1
    haus_coeff = -1
    try:
        dice_coeff = slicer.util.getNode("Dice*").GetTable().GetRow(4).GetValue(1).ToDouble()
        haus_coeff = slicer.util.getNode("Haus*").GetTable().GetRow(4).GetValue(1).ToDouble()
    except Exception:
        print("Could not fetch coefficients")
    
    return node_pre_trans, node_def_trans, dice_coeff, haus_coeff


def process_batch_patients(folder_mr, folder_us, folder_trans):
    folder_trans.mkdir(exist_ok=True)

    for patient_folder_mr in folder_mr.iterdir():
        patient_folder_us = folder_us / f"{patient_folder_mr.name}"
        if not patient_folder_us.exists():
            print("No related US folder exists, skipping")
            continue
        patient_folder_trans = folder_trans / f"{patient_folder_mr.name}"
        if patient_folder_trans.exists():
            print(f"Skipping existing patient")
            continue
        patient_folder_trans.mkdir(exist_ok=True)
        process_patient(patient_folder_mr, patient_folder_us, patient_folder_trans)


def process_patient(patient_folder_mr, patient_folder_us, patient_folder_trans):
    patient_name = patient_folder_mr.name
    
    # iterate over the MRIs
    for folder_mr_data in patient_folder_mr.iterdir():
        # fail-safe MRI load
        try:
            mr_image = list(folder_mr_data.rglob("MRI*"))[0]
            mr_prostate = list(folder_mr_data.rglob("Prostate*"))[0]
        except Exception as ex:
            print(f"Cannot load MRI {folder_mr_data.name} for {patient_name}, skipping this MRI")
            print(ex)
            continue
        
        # iterate of the USs
        for folder_us_data in patient_folder_us.iterdir():
            # fail-safe US load
            try:
                us_image = list(folder_us_data.rglob("US*"))[0]
                us_prostate = list(folder_us_data.rglob("Prostate*"))[0]
            except Exception as ex:
                print(f"Cannot load US {folder_us_data.name} for {patient_name}, skipping this US")
                print(ex)
                continue
            
            # perform processing
            print(f"Processing {patient_name} over MRI/US combination {folder_mr_data.name}/{folder_us_data.name} ...")
            node_pre_trans, node_def_trans, dice_coeff, haus_coeff = process(str(mr_image), str(mr_prostate), str(us_image), str(us_prostate))
            print("... saving ...")
            pre_trans_file = str(patient_folder_trans / f"pre_{folder_mr_data.name}_{folder_us_data.name}.h5")
            def_trans_file = str(patient_folder_trans / f"def_{folder_mr_data.name}_{folder_us_data.name}.h5")
            json_file = str(patient_folder_trans / f"coeffs_{folder_mr_data.name}_{folder_us_data.name}.json")
            json_content = {"dice": dice_coeff, "hausdorff": haus_coeff}
            print(json_content)
            with open(json_file, 'w') as f:
                json.dump(json_content, f)
            slicer.util.saveNode(node_pre_trans, pre_trans_file, properties={'useCompression': 1})
            slicer.util.saveNode(node_def_trans, def_trans_file, properties={'useCompression': 1})
            print("done")


def main_test_process():
    path_mr_image = r"C:\Users\sebas\Documents\Tesi\nrrd\paziente1_mri\Dato1\MRI1_1.3.6.1.4.1.14519.5.2.1.266717969984343981963002258381778490221.nrrd"
    path_mr_seg = r"C:\Users\sebas\Documents\Tesi\nrrd\paziente1_mri\Dato1\Prostate1_1.3.6.1.4.1.14519.5.2.1.266717969984343981963002258381778490221.nrrd"
    path_us_image = r"C:\Users\sebas\Documents\Tesi\nrrd\paziente1_us\Dato1\US_1.3.6.1.4.1.14519.5.2.1.140367896789002601449386011052978380612.nrrd"
    path_us_seg = r"C:\Users\sebas\Documents\Tesi\nrrd\paziente1_us\Dato1\Prostate_1.3.6.1.4.1.14519.5.2.1.140367896789002601449386011052978380612.nrrd"
    process(path_mr_image, path_mr_seg, path_us_image, path_us_seg)


if __name__ == "__main__":
    try:
        folder_base = Path(r"R:\DATASET_PROSTATE")
        folder_mr = folder_base / "mri"
        folder_us = folder_base / "us"
        folder_trans = folder_base / "reg"
        process_batch_patients(folder_mr, folder_us, folder_trans)
    except Exception as ex:
        print(ex)
