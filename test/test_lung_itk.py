import itk
import numpy as np
import unittest
import matplotlib.pyplot as plt
import numpy as np

import icon_registration.test_utils
import icon_registration.pretrained_models
import icon_registration.itk_wrapper


class TestItkRegistration(unittest.TestCase):
    def test_itk_registration(self):

        model = icon_registration.pretrained_models.LungCT_registration_model(
            pretrained=True
        )
        
        icon_registration.test_utils.download_test_data()

        image_exp = itk.imread(
            str(
                icon_registration.test_utils.TEST_DATA_DIR
                / "lung_test_data/copd1_highres_EXP_STD_COPD_img.nii.gz"
            )
        )
        image_insp = itk.imread(
            str(
                icon_registration.test_utils.TEST_DATA_DIR
                / "lung_test_data/copd1_highres_INSP_STD_COPD_img.nii.gz"
            )
        )
        image_exp_seg = itk.imread(
            str(
                icon_registration.test_utils.TEST_DATA_DIR
                / "lung_test_data/copd1_highres_EXP_STD_COPD_label.nii.gz"
            )
        )
        image_insp_seg = itk.imread(
            str(
                icon_registration.test_utils.TEST_DATA_DIR
                / "lung_test_data/copd1_highres_INSP_STD_COPD_label.nii.gz"
            )
        )

        image_insp_preprocessed = (
            icon_registration.pretrained_models.lung_network_preprocess(
                image_insp, image_insp_seg
            )
        )
        image_exp_preprocessed = (
            icon_registration.pretrained_models.lung_network_preprocess(
                image_exp, image_exp_seg
            )
        )

        phi_AB, phi_BA = icon_registration.itk_wrapper.register_pair(
            model, image_insp_preprocessed, image_exp_preprocessed
        )

        assert isinstance(phi_AB, itk.CompositeTransform)
        interpolator = itk.LinearInterpolateImageFunction.New(image_insp_preprocessed)

        warped_image_insp_preprocessed = itk.resample_image_filter(
            image_insp_preprocessed,
            transform=phi_AB,
            interpolator=interpolator,
            size=itk.size(image_exp_preprocessed),
            output_spacing=itk.spacing(image_exp_preprocessed),
            output_direction=image_exp_preprocessed.GetDirection(),
            output_origin=image_exp_preprocessed.GetOrigin(),
        )

        # log some images to show the registration
        import os
        os.environ["FOOTSTEPS_NAME"] = "test"
        import footsteps

        plt.imshow(
            np.array(
                itk.checker_board_image_filter(
                    warped_image_insp_preprocessed, image_exp_preprocessed
                )
            )[140]
        )
        plt.colorbar()
        plt.savefig(footsteps.output_dir + "grid_lung.png")
        plt.clf()
        plt.imshow(np.array(warped_image_insp_preprocessed)[140])
        plt.colorbar()
        plt.savefig(footsteps.output_dir + "warped_lung.png")
        plt.clf()
        plt.imshow(
            np.array(warped_image_insp_preprocessed)[140]
            - np.array(image_exp_preprocessed)[140]
        )
        plt.colorbar()
        plt.savefig(footsteps.output_dir + "difference_lung.png")
        plt.clf()

        insp_points = read_copd_pointset(
            "test_files/lung_test_data/copd1_300_iBH_xyz_r1.txt"
        )
        exp_points = read_copd_pointset(
            "test_files/lung_test_data/copd1_300_eBH_xyz_r1.txt"
        )
        dists = []
        for i in range(len(insp_points)):
            px, py = (
                exp_points[i],
                np.array(phi_BA.TransformPoint(tuple(insp_points[i]))),
            )
            dists.append(np.sqrt(np.sum((px - py) ** 2)))
        print(np.mean(dists))

        self.assertLess(np.mean(dists), 1.5)
        dists = []
        for i in range(len(insp_points)):
            px, py = (
                insp_points[i],
                np.array(phi_AB.TransformPoint(tuple(exp_points[i]))),
            )
            dists.append(np.sqrt(np.sum((px - py) ** 2)))
        print(np.mean(dists))
        self.assertLess(np.mean(dists), 2.3)

COPD_spacing = {
    "copd1": [0.625, 0.625, 2.5],
    "copd2": [0.645, 0.645, 2.5],
    "copd3": [0.652, 0.652, 2.5],
    "copd4": [0.590, 0.590, 2.5],
    "copd5": [0.647, 0.647, 2.5],
    "copd6": [0.633, 0.633, 2.5],
    "copd7": [0.625, 0.625, 2.5],
    "copd8": [0.586, 0.586, 2.5],
    "copd9": [0.664, 0.664, 2.5],
    "copd10": [0.742, 0.742, 2.5],
}


def read_copd_pointset(f_path):
    """
    :param f_path: the path to the file containing the position of points.
    Points are deliminated by '\n' and X,Y,Z of each point are deliminated by '\t'.
    :return: numpy array of points in physical coordinates
    """
    spacing = COPD_spacing[f_path.split("/")[-1].split("_")[0]]
    spacing = np.expand_dims(spacing, 0)
    with open(f_path) as fp:
        content = fp.read().split("\n")

        # Read number of points from second
        count = len(content) - 1

        # Read the points
        points = np.ndarray([count, 3], dtype=np.float64)
        for i in range(count):
            if content[i] == "":
                break
            temp = content[i].split("\t")
            points[i, 0] = float(temp[0])
            points[i, 1] = float(temp[1])
            points[i, 2] = float(temp[2])

        # The copd gene points are in index space instead of physical space.
        # Move them to physical space.
        return (points - 1) * spacing
