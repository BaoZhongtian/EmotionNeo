from Auxiliary.Loader import LoadSpectrumWlabel
from Model.BLSTMwAttention import BLSTMwAttention
from Auxiliary.TrainTemplate import TrainTemplate_Basic

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
                                                  savePath='D:/PythonProjects_DataNeo/Result/%s-%s/%s-%d' % (
                                                      appointMedia, attentionName, appointGender, appointSession))
                TrainMethod.TrainTestProgress()
