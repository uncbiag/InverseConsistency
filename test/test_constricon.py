import unittest

class TestConstricont(unittest.TestCase):
    def test_2d_inverse_consistent_train(self):
    
        import icon_registration
        import icon_registration.data as data
        import icon_registration.networks as networks
        from icon_registration import constricon
        from icon_registration import SSD

        import numpy as np
        import torch
        import random
        import os

        random.seed(1)
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        np.random.seed(1)

        batch_size = 128

        d1, d2 = data.get_dataset_triangles(
            data_size=50, hollow=False, batch_size=batch_size
        )
        d1_t, d2_t = data.get_dataset_triangles(
            data_size=50, hollow=False, batch_size=batch_size
        )


        print("ConstrICON training")
        net = constricon.VelocityFieldDiffusion(
            constricon.FirstTransform(
                constricon.TwoStepInverseConsistent(
                    constricon.ConsistentFromMatrix(
                        networks.ConvolutionalMatrixNet(dimension=2)
                    ), 
                    constricon.ICONSquaringVelocityField(
                        networks.tallUNet2(dimension=2)
                    )
                ),
            ),
            SSD(),
            3,
        )

        input_shape = list(next(iter(d1))[0].size())
        input_shape[0] = 1
        net.assign_identity_map(input_shape)
        net.cuda()
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        net.train()

        y = icon_registration.train_datasets(net, optimizer, d1, d2, epochs=5)

        # Test that image similarity is good enough
        self.assertLess(np.mean(np.array(y)[-5:, 1]), 0.1)

        # Test that folds are rare enough
        self.assertLess(np.mean(np.exp(np.array(y)[-5:, 3] - 0.1)), 2)
        for l in y[:-3]:
            print(l)
