import unittest
from weight_initializer import UniformWeightInitializer
from weight import Weight

class WeightTest(unittest.TestCase):
    def testSerialization(self):
        weight = Weight(name='name1',
                        initializer=UniformWeightInitializer(
                            limit=(-1, 1),
                            seed=2
                        ))
        weight.initialize(shape=(10, 10))
        wser = weight.serialize()
        self.assertTrue('name1' in wser)
        self.assertTrue(hasattr(wser['name1'], 'shape'))
        self.assertTrue(wser['name1'].shape, (10, 10))
        wdict = weight.toDict()
        self.assertTrue('name' in wdict)
        self.assertEqual(wdict['name'], 'name1')
        self.assertTrue('initializerSpec' in wdict)
        self.assertEqual(wdict['initializerSpec']['limit'], '(-1, 1)')
        self.assertEqual(wdict['initializerSpec']['seed'], 2)
        self.assertEqual(wdict['initializerSpec']['type'], 'uniform')
        self.assertTrue('shared' in wdict)
        self.assertEqual(wdict['shared'], False)
        self.assertTrue('gpuEnabled' in wdict)
        self.assertEqual(wdict['gpuEnabled'], False)

        weight2 = Weight.fromDict(wdict)
        weight2.initialize((10, 10))
        wnumpy1 = weight.get()
        wnumpy2 = weight2.get()
        for i in range(10):
            for j in range(10):
                self.assertEqual(wnumpy1[i, j], wnumpy2[i, j])
        self.assertEqual(weight2.gpuEnabled, False)
        self.assertEqual(weight2.shared, False)
        pass

if __name__ == '__main__':
    unittest.main()
