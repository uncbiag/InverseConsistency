import icon_registration as icon
import cvpr_network
import torch
import itk
import icon_registration.itk_wrapper as itk_wrapper
import dice


input_shape = [1, 1, 80, 192, 192]
net = cvpr_network.make_network(input_shape, include_last_step=True, lmbda=.2, loss_fn=icon.ssd_only_interpolated)

weights_path = "/playpen-raid1/tgreer/ICON/training_scripts/gradICON/results/ent2end_thenonemore-7/network_weights_26400"

net.regis_net.load_state_dict(torch.load(weights_path))


with open("../oai_paper_pipeline/splits/test/pair_name_list.txt") as f:
    test_pair_names = f.readlines()
with open("../oai_paper_pipeline/splits/test/pair_path_list.txt") as f:
    test_pair_paths = f.readlines()

dices = []

for test_pair in test_pair_paths:
    test_pair = test_pair.replace("playpen", "playpen-raid").split()
    test_pair = [itk.imread(path) for path in test_pair]
    image_A, image_B, segmentation_A, segmentation_B = test_pair

    phi_AB, phi_BA = itk_wrapper.register_pair(net, image_A, image_B)

    interpolator = itk.NearestNeighborInterpolateImageFunction.New(segmentation_A)

    warped_segmentation_A = itk.resample_image_filter(
            segmentation_A, 
            transform=phi_AB,
            interpolator=interpolator,
            use_reference_image=True,
            reference_image=segmentation_B
            )
    mean_dice = dice.itk_mean_dice(segmentation_B, warped_segmentation_A)

    print(mean_dice)

    dices.append(mean_dice)

print("Mean DICE")
print(np.mean(dices))
