import glob
import os

import numpy as np

# IMPORTANT: use Cellpose's imread so that .npy "_seg.npy" files are handled
# exactly like upstream (np.load + ['masks'] extraction). Do NOT replace with
# tifffile.imread here or .npy will fail with a TiffFileError.
from cellpose.io import imread
from loguru import logger
from natsort import natsorted

_MASK_FALLBACK_SUFFIXES = ("_masks.npy", "_masks.tif", "_masks.tiff", "_masks.png")


def get_image_files(folder, mask_filter, imf=None, look_one_level_down=False):
    """
    Finds all images in a folder and its subfolders (if specified) with the given file extensions.

    Args:
        folder (str): The path to the folder to search for images.
        mask_filter (str): The filter for mask files.
        imf (str, optional): The additional filter for image files. Defaults to None.
        look_one_level_down (bool, optional): Whether to search for images in subfolders. Defaults to False.

    Returns:
        list: A list of image file paths.

    Raises:
        ValueError: If no files are found in the specified folder.
        ValueError: If no images are found in the specified folder with the supported file extensions.
        ValueError: If no images are found in the specified folder without the mask or flow file endings.
    """
    mask_filters = [
        "_cp_output",
        "_flows",
        "_flows_0",
        "_flows_1",
        "_flows_2",
        "_cellprob",
        "_masks",
        mask_filter,
    ]
    image_names = []
    if imf is None:
        imf = ""

    folders = []
    if look_one_level_down:
        folders = natsorted(glob.glob(os.path.join(folder, "*/")))
    folders.append(folder)
    exts = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".flex", ".dax", ".nd2", ".nrrd"]
    l0 = 0
    al = 0
    for folder in folders:
        all_files = glob.glob(folder + "/*")
        al += len(all_files)
        for ext in exts:
            image_names.extend(glob.glob(folder + f"/*{imf}{ext}"))
            image_names.extend(glob.glob(folder + f"/*{imf}{ext.upper()}"))
        l0 += len(image_names)

    # return error if no files found
    if al == 0:
        return []
    elif l0 == 0:
        return []

    image_names = natsorted(image_names)
    imn = []
    for im in image_names:
        imfile = os.path.splitext(im)[0]
        igood = all([
            (len(imfile) > len(mask_filter) and imfile[-len(mask_filter) :] != mask_filter)
            or len(imfile) <= len(mask_filter)
            for mask_filter in mask_filters
        ])
        if len(imf) > 0:
            igood &= imfile[-len(imf) :] == imf
        if igood:
            imn.append(im)

    image_names = imn

    # remove duplicates
    image_names = [*set(image_names)]
    image_names = natsorted(image_names)

    if len(image_names) == 0:
        raise ValueError("ERROR: no images in --dir folder without _masks or _flows or _cellprob ending")

    return image_names


def get_label_files(image_names, mask_filter, imf=None):
    """
    Get the label files corresponding to the given image names and mask filter.

    Args:
        image_names (list): List of image names.
        mask_filter (str): Mask filter to be applied.
        imf (str, optional): Image file extension. Defaults to None.

    Returns:
        tuple: A tuple containing the label file names and flow file names (if present).
    """
    nimg = len(image_names)
    label_names0 = [os.path.splitext(image_names[n])[0] for n in range(nimg)]

    if not label_names0:
        return [], None

    if imf is not None and len(imf) > 0:
        label_names = [label_names0[n][: -len(imf)] for n in range(nimg)]
    else:
        label_names = label_names0

    # check for flows
    if os.path.exists(label_names0[0] + "_flows.tif"):
        flow_names = [label_names0[n] + "_flows.tif" for n in range(nimg)]
    else:
        flow_names = [label_names[n] + "_flows.tif" for n in range(nimg)]
    if not all([os.path.exists(flow) for flow in flow_names]):
        logger.info("not all flows are present, running flow generation for all images")
        flow_names = None

    # check for masks
    if mask_filter == "_seg.npy":
        resolved_labels: list[str] = []
        for base_name in label_names:
            seg_candidate = base_name + mask_filter
            if os.path.exists(seg_candidate):
                resolved_labels.append(seg_candidate)
                continue

            fallback = None
            for suffix in _MASK_FALLBACK_SUFFIXES:
                candidate = base_name + suffix
                if os.path.exists(candidate):
                    fallback = candidate
                    break

            resolved_labels.append(fallback if fallback is not None else seg_candidate)

        return resolved_labels, None

    if os.path.exists(label_names[0] + mask_filter + ".tif"):
        label_names = [label_names[n] + mask_filter + ".tif" for n in range(nimg)]
    elif os.path.exists(label_names[0] + mask_filter + ".tiff"):
        label_names = [label_names[n] + mask_filter + ".tiff" for n in range(nimg)]
    elif os.path.exists(label_names[0] + mask_filter + ".png"):
        label_names = [label_names[n] + mask_filter + ".png" for n in range(nimg)]
    # TODO, allow _seg.npy
    # elif os.path.exists(label_names[0] + "_seg.npy"):
    #    io_logger.info("labels found as _seg.npy files, converting to tif")
    else:
        if not flow_names:
            raise ValueError("labels not provided with correct --mask_filter")
        else:
            label_names = None
    if not all([os.path.exists(label) for label in label_names]):
        if not flow_names:
            raise ValueError("labels not provided for all images in train and/or test set")
        else:
            label_names = None

    return label_names, flow_names


def load_images_labels(tdir, mask_filter="_masks", image_filter=None, look_one_level_down=False):
    """
    Loads images and corresponding labels from a directory.

    Args:
        tdir (str): The directory path.
        mask_filter (str, optional): The filter for mask files. Defaults to "_masks".
        image_filter (str, optional): The filter for image files. Defaults to None.
        look_one_level_down (bool, optional): Whether to look for files one level down. Defaults to False.

    Returns:
        tuple: A tuple containing a list of images, a list of labels, and a list of image names.
    """
    image_names = get_image_files(tdir, mask_filter, image_filter, look_one_level_down)
    nimg = len(image_names)

    # training data
    label_names, flow_names = get_label_files(image_names, mask_filter, imf=image_filter)

    images = []
    labels = []
    # Align names to actually-loaded items: upstream returns the original
    # image_names even when some entries are skipped (no masks/flows), which
    # causes a length mismatch (len(images/labels) != len(image_names)).
    # We track kept_image_names and return that instead to keep lengths equal.
    kept_image_names = []
    for n in range(nimg):
        # Only keep samples that have either a mask file or (if configured)
        # a corresponding flow file. This mirrors upstream selection logic
        # but additionally records the kept image name for alignment.
        if os.path.isfile(label_names[n]) or (flow_names is not None and os.path.isfile(flow_names[0])):
            image = imread(image_names[n])
            if label_names is not None:
                label = imread(label_names[n])
            if flow_names is not None:
                flow = imread(flow_names[n])
                if flow.shape[0] < 4:
                    label = np.concatenate((label[np.newaxis, :, :], flow), axis=0)
                else:
                    label = flow
            images.append(image)
            labels.append(label)
            kept_image_names.append(image_names[n])
    logger.info(f"{len(images)} / {nimg} images in {tdir} folder have labels")
    return images, labels, kept_image_names


def load_train_test_data(
    train_dir, test_dir=None, image_filter=None, mask_filter="_masks", look_one_level_down=False
):
    """
    Loads training and testing data for a Cellpose model.

    Args:
        train_dir (str): The directory path containing the training data.
        test_dir (str, optional): The directory path containing the testing data. Defaults to None.
        image_filter (str, optional): The filter for selecting image files. Defaults to None.
        mask_filter (str, optional): The filter for selecting mask files. Defaults to "_masks".
        look_one_level_down (bool, optional): Whether to look for data in subdirectories of train_dir and test_dir. Defaults to False.

    Returns:
        images, labels, image_names, test_images, test_labels, test_image_names

    """
    images, labels, image_names = load_images_labels(
        train_dir, mask_filter, image_filter, look_one_level_down
    )
    # testing data
    test_images, test_labels, test_image_names = None, None, None
    if test_dir is not None:
        test_images, test_labels, test_image_names = load_images_labels(
            test_dir, mask_filter, image_filter, look_one_level_down
        )

    return images, labels, image_names, test_images, test_labels, test_image_names
