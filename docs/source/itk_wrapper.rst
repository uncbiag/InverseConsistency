ICON and ITK
====================

Often, a downstream clinical registration task requires warping landmarks, or warping images of different physical spacings in physical coordinate systems. ITK is a medical imaging library with excellent, correct handling of images with assosciated coordinate systems, and ICON integrates with ITK to make registration in physical coordinates convenient.

To demonstrate this, we will register a pair of lung images from the COPD gene dataset, and then use the resulting transform to move landmarks in physical coordinates.

`notebook <https://github.com/name/ICON/blob/master/notebooks/ICON_lung_demo.ipynb>`_

Setup
-----

.. plot::
   :include-source:
   :context:

   import itk
   import numpy as np
   import unittest
   import matplotlib.pyplot as plt
   import numpy as np

   import icon_registration.pretrained_models
   import icon_registration.itk_wrapper
   import icon_registration.test_utils
   icon_registration.test_utils.download_test_data()

   model = icon_registration.pretrained_models.LungCT_registration_model(
       pretrained=True
   )

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
       model, image_insp_preprocessed, image_exp_preprocessed, finetune_steps=None
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

.. plot::
   :context:
   :include-source:

   plt.imshow(
       np.array(
           itk.checker_board_image_filter(
               warped_image_insp_preprocessed, image_exp_preprocessed
           )
       )[140]
   )
   plt.colorbar()

.. plot::
   :context:
   :include-source:

   plt.close()
   plt.imshow(np.array(warped_image_insp_preprocessed)[140])
   plt.colorbar()

.. plot::
   :context:
   :include-source:
   
   plt.close()
   plt.imshow(
       np.array(warped_image_insp_preprocessed)[140]
       - np.array(image_exp_preprocessed)[140]
   )
   plt.colorbar()


To move physical points, use phi_BA.TransformPoint

.. plot::
   :context:
   :include-source:

   insp_points = icon_registration.test_utils.read_copd_pointset(
           str(icon_registration.test_utils.TEST_DATA_DIR)
       + "/lung_test_data/copd1_300_iBH_xyz_r1.txt"
   )
   exp_points = icon_registration.test_utils.read_copd_pointset(
           str(icon_registration.test_utils.TEST_DATA_DIR)
       + "/lung_test_data/copd1_300_eBH_xyz_r1.txt"
   )

   warped_insp_points = []
   for i in range(len(insp_points)):
       px, py = (
           exp_points[i],
           np.array(phi_BA.TransformPoint(tuple(insp_points[i]))),
       )
       warped_insp_points.append(py)
   warped_insp_points = np.array(warped_insp_points)

   plt.close()
   def scatxy(pts):
      plt.scatter(pts[:, 0], pts[:, 1])

   scatxy(insp_points)
   scatxy(exp_points)

.. plot::
   :context:
   :include-source:

   plt.close()
   scatxy(warped_insp_points)
   scatxy(exp_points)


.. plot::
   :nofigs:
   :context:

   plt.close()
