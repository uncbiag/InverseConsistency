import footsteps
import itk
import ants
footsteps.initialize(output_root="evaluation_results/")
import numpy as np
import utils


with open("../oai_paper_pipeline/splits/test/pair_path_list.txt") as f:
    test_pair_paths = f.readlines()

dices = []

def mean_dice_f(sA, sB):
    sA, sB = [itk.image_from_array(s.numpy()) for s in (sA, sB)]
    return utils.itk_mean_dice(sA, sB)

for test_pair_path in test_pair_paths:
    test_pair_path = test_pair_path.replace("author", "author-raid").split()
    test_pair = [itk.imread(path) for path in test_pair_path]
    test_pair = [
            (itk.flip_image_filter(t, flip_axes=(False, False, True))
                if "RIGHT" in path else t 
                ) for (t , path) in zip(test_pair, test_pair_path)]

    test_pair = [ants.from_numpy(itk.array_from_image(t)) for t in test_pair]

    

    image_A, image_B, segmentation_A, segmentation_B = test_pair

    reg_res = ants.registration(image_A, image_B, "SyNAggro", outprefix=footsteps.output_dir)
    warped_segmentation_B = ants.apply_transforms(fixed=segmentation_A, moving=segmentation_B, transformlist=reg_res['fwdtransforms'], interpolator="nearestNeighbor")

    mean_dice = mean_dice_f(segmentation_A, warped_segmentation_B)

    utils.log(mean_dice)

    dices.append(mean_dice)

utils.log("Mean DICE")
utils.log(np.mean(dices))
