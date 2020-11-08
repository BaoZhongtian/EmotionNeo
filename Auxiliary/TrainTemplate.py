import os
import torch
import tqdm
import numpy


class TrainTemplate_Base():
    def __init__(self, Model, trainDataset, testDataset, trainEpoch, learningRate, optimizerType='Adam', saveFlag=False,
                 savePath=None, cudaFlag=False):
        self.Model = Model
        self.trainDataset, self.testDataset = trainDataset, testDataset
        self.trainEpoch, self.learningRate = trainEpoch, learningRate
        self.saveFlag, self.savePath = saveFlag, savePath
        self.cudaFlag = cudaFlag
        if self.cudaFlag: self.Model.cuda()

        self.optimizer = None
        if optimizerType == 'Adam':
            self.optimizer = torch.optim.Adam(params=self.Model.parameters(), lr=learningRate)

        self.EndFlag = False
        self.Pretreatment()

    def Pretreatment(self):
        if self.saveFlag:
            if os.path.exists(self.savePath):
                self.EndFlag = True
                return
            os.makedirs(self.savePath)
            os.makedirs(self.savePath + '-TestResult')
        if self.optimizer is None:
            raise RuntimeError('Please Give a Correct Optimizer Name')

    def SaveParameter(self, epochNumber):
        torch.save(obj={'ModelStateDict': self.Model.state_dict(), 'OptimizerStateDict': self.optimizer.state_dict()},
                   f=os.path.join(self.savePath, 'Parameter-%04d.pkl' % epochNumber))
        torch.save(obj=self.Model, f=os.path.join(self.savePath, 'Network-%04d.pkl' % epochNumber))

    def TrainTestProgress(self):
        if self.EndFlag:
            print('Fold Already Existed.', self.savePath)
            return
        for index in range(self.trainEpoch):
            self.Model.train()
            loss = self.TrainEpoch(epochNumber=index)
            print('Episode %d : Total Loss = %f' % (index, loss))
            self.Model.eval()
            self.TestEpoch(epochNumber=index)
            self.SaveParameter(epochNumber=index)

    def TrainDataSelection(self, batchData):
        return batchData

    def TrainEpoch(self, epochNumber):
        totalLoss = 0.0
        if self.saveFlag: file = open(os.path.join(self.savePath, 'Loss-%04d.csv' % epochNumber), 'w')
        for batchData in tqdm.tqdm(self.trainDataset):
            batchData = self.TrainDataSelection(batchData)
            loss = self.Model(batchData)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            lossValue = loss.detach().cpu().numpy()
            totalLoss += lossValue
            if self.saveFlag: file.write(str(lossValue) + '\n')
        if self.saveFlag: file.close()
        return totalLoss

    def TestDataSelection(self, batchData):
        return batchData

    def TestEpoch(self, epochNumber):
        if self.saveFlag: file = open(os.path.join(self.savePath + '-TestResult', 'Predict-%04d.csv' % epochNumber),
                                      'w')
        for batchData in self.testDataset:
            batchDataTreated = self.TestDataSelection(batchData)
            predict = self.Model(batchDataTreated)

            predict = predict.detach().cpu().numpy()
            result = numpy.concatenate([predict, batchData[-1].numpy().reshape([-1, 1])], axis=1)

            if self.saveFlag:
                for indexX in range(numpy.shape(result)[0]):
                    for indexY in range(numpy.shape(result)[1]):
                        if indexY != 0: file.write(',')
                        file.write(str(result[indexX][indexY]))
                    file.write('\n')
        if self.saveFlag: file.close()

    def LoadProgress(self, loadPath, testDataFlag=False):
        checkpoint = torch.load(loadPath)
        self.Model.load_state_dict(checkpoint['ModelStateDict'])
        self.optimizer.load_state_dict(checkpoint['OptimizerStateDict'])
        if testDataFlag: self.TestEpoch(epochNumber=9999)


class TrainTemplate_Basic(TrainTemplate_Base):
    def __init__(self, Model, trainDataset, testDataset, trainEpoch, learningRate, optimizerType='Adam', saveFlag=False,
                 savePath=None, cudaFlag=False):
        super(TrainTemplate_Basic, self).__init__(Model, trainDataset, testDataset, trainEpoch, learningRate,
                                                  optimizerType, saveFlag, savePath, cudaFlag)

    def TrainDataSelection(self, batchData):
        return {'inputData': batchData[0], 'inputSeqLen': batchData[1], 'inputLabel': batchData[2]}

    def TestDataSelection(self, batchData):
        return {'inputData': batchData[0], 'inputSeqLen': batchData[1]}


class TrainTemplate_AttentionTransform(TrainTemplate_Base):
    def __init__(self, ModelTarget, ModelSource, trainDataset, testDataset, trainEpoch, learningRate,
                 transformModelPath, sourceMedia, optimizerType='Adam', saveFlag=False, savePath=None, cudaFlag=False):
        self.ModelSource = ModelSource
        self.LoadSourceModel(loadPath=transformModelPath)
        self.sourceMedia = sourceMedia

        super(TrainTemplate_AttentionTransform, self).__init__(
            ModelTarget, trainDataset, testDataset, trainEpoch, learningRate, optimizerType, saveFlag, savePath,
            cudaFlag)
        if self.cudaFlag: self.ModelSource.cuda()

    def TrainDataSelection(self, batchData):
        if self.sourceMedia == 'Audio':
            attentionInput = {'inputData': batchData[0], 'inputSeqLen': batchData[1], 'attentionFlag': True}
        if self.sourceMedia == 'Video':
            attentionInput = {'inputData': batchData[2], 'inputSeqLen': batchData[3], 'attentionFlag': True}
        attentionResult = self.ModelSource(attentionInput)

        if self.sourceMedia == 'Audio':
            return {'inputData': batchData[2], 'inputSeqLen': batchData[3], 'inputLabel': batchData[4],
                    'inputAttentionMap': attentionResult, 'inputAttentionSeq': batchData[1]}
        if self.sourceMedia == 'Video':
            return {'inputData': batchData[0], 'inputSeqLen': batchData[1], 'inputLabel': batchData[4],
                    'inputAttentionMap': attentionResult, 'inputAttentionSeq': batchData[3]}
        raise RuntimeError('Please Give Correct Media')

    def TestDataSelection(self, batchData):
        if self.sourceMedia == 'Audio':
            return {'inputData': batchData[2], 'inputSeqLen': batchData[3], 'inputLabel': batchData[4]}
        if self.sourceMedia == 'Video':
            return {'inputData': batchData[0], 'inputSeqLen': batchData[1], 'inputLabel': batchData[4]}

    def LoadSourceModel(self, loadPath):
        checkpoint = torch.load(loadPath)
        self.ModelSource.load_state_dict(checkpoint['ModelStateDict'])


if __name__ == '__main__':
    from Auxiliary.Loader import LoadSpectrumWAudioVideo
    from Model.BLSTMwAttention import BLSTMwAttention, BLSTMwAttention_AttentionTransform

    appointMedia = 'Audio'
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
