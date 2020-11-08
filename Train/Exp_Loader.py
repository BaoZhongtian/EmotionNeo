from Auxiliary.Loader import LoadSpectrumWlabel
from Model.BLSTMwAttention import BLSTMwAttention
from Auxiliary.TrainTemplate import TrainTemplate_Basic
from Auxiliary.Tools import LoadNetwork
import torch

if __name__ == '__main__':
    appointMedia = 'Audio'
    for attentionName in ['StandardAttention', 'LocalAttention', 'ComponentAttention', 'MonotonicAttention']:
        for appointGender in ['Female', 'Male']:
            for appointSession in range(1, 6):
                trainDataset, testDataset = LoadSpectrumWlabel(
                    appointMedia=appointMedia, appointGender=appointGender, appointSession=appointSession)
                model = BLSTMwAttention(attentionName=attentionName, attentionScope=10, featuresNumber=40,
                                        classNumber=4, cudaFlag=True)
                TrainMethod = TrainTemplate_Basic(Model=model, trainDataset=trainDataset, testDataset=testDataset,
                                                  trainEpoch=100, learningRate=1E-3, saveFlag=True, cudaFlag=True,
                                                  savePath='Result/%s-%s/%s-%d' % (
                                                      appointMedia, attentionName, appointGender, appointSession))
                TrainMethod.LoadProgress('D:/PythonFiles/EmotionNeo/Result-SingleMedia-Parameter/%s-%s/%s-%d.pkl' % (
                    appointMedia, attentionName, appointGender, appointSession), testDataFlag=True)
                exit()
