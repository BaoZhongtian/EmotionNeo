from Auxiliary.Loader import LoadSpectrumWAudioVideo
from Auxiliary.TrainTemplate import TrainTemplate_AttentionTransform
from Model.BLSTMwAttention import BLSTMwAttention, BLSTMwAttention_AttentionTransform

if __name__ == '__main__':
    appointMedia = 'Video'
    cudaFlag = True
    transformWeight = 1E+4
    for attentionName in ['StandardAttention', 'LocalAttention', 'ComponentAttention', 'MonotonicAttention']:
        for appointGender in ['Female', 'Male']:
            for appointSession in range(1, 6):
                trainDataset, testDataset = LoadSpectrumWAudioVideo(
                    appointGender=appointGender, appointSession=appointSession)
                transformPath = 'D:/PythonFiles/EmotionNeo/Result-SingleMedia-Parameter/%s-%s/%s-%d.pkl' % (
                    appointMedia, attentionName, appointGender, appointSession)
                savePath = 'TransformFrom%s-%s/%s-%d' % (
                    appointMedia, attentionName, appointGender, appointSession)

                if appointMedia == 'Video':
                    modelTarget = BLSTMwAttention_AttentionTransform(
                        attentionName=attentionName, attentionScope=10, featuresNumber=40, classNumber=4,
                        cudaFlag=cudaFlag, transformWeight=transformWeight)
                    modelSource = BLSTMwAttention(attentionName=attentionName, attentionScope=10, featuresNumber=175,
                                                  classNumber=4, cudaFlag=cudaFlag)
                if appointMedia == 'Audio':
                    modelTarget = BLSTMwAttention_AttentionTransform(
                        attentionName=attentionName, attentionScope=10, featuresNumber=175, classNumber=4,
                        cudaFlag=cudaFlag, transformWeight=transformWeight)
                    modelSource = BLSTMwAttention(attentionName=attentionName, attentionScope=10, featuresNumber=40,
                                                  classNumber=4, cudaFlag=cudaFlag)
                TrainMethod = TrainTemplate_AttentionTransform(
                    modelTarget, modelSource, trainDataset, testDataset, sourceMedia=appointMedia, trainEpoch=100,
                    learningRate=1E-3, transformModelPath=transformPath, saveFlag=True,
                    savePath=savePath, cudaFlag=cudaFlag)
                TrainMethod.TrainTestProgress()
