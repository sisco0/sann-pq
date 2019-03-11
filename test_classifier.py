import unittest
from classifier import classifier
import numpy as np

class TestClassifier(unittest.TestCase):

    classifier_instance = classifier()
    def SetUp(self):
        pass

    def test_instantation(self):
        print(self.classifier_instance.model.summary())

    def test_fitting(self):
        time = np.array([k/self.classifier_instance.sf
            for k in range(self.classifier_instance.tp)])
        for k in range(10):
            signal = np.sin(2.0*np.pi*60.0*time)
            st = random.randrange(0, 600)
            signal[st:st+300] = (0.75+k/100)*signal[st:st+300]
            signal = np.reshape(signal,
                (1,self.classifier_instance.tp, 1))
            print('Signal ', 0.75+k/100, 'pu: ',
                self.classifier_instance.model.predict(signal))

if __name__ == '__main__':
    unittest.main()
