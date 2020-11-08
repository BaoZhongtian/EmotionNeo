import torch
import numpy
from Model.AttentionBase import AttentionBase


class BLSTMwAttention(AttentionBase):
    def __init__(self, attentionName, attentionScope, featuresNumber, classNumber, cudaFlag):
        super(BLSTMwAttention, self).__init__(
            attentionName=attentionName, attentionScope=attentionScope, featuresNumber=128 * 2, cudaFlag=cudaFlag)
        self.moduleName = 'BLSTM-W-%s-%d' % (attentionName, attentionScope)
        self.rnnLayer = torch.nn.LSTM(input_size=featuresNumber, hidden_size=128, num_layers=2, bidirectional=True)
        self.predict = torch.nn.Linear(in_features=256, out_features=classNumber)
        self.lossFunction = torch.nn.CrossEntropyLoss()

    def forward(self, batchData):
        inputData = batchData['inputData']
        inputSeqLen = batchData['inputSeqLen']
        if 'inputLabel' in batchData.keys():
            inputLabel = batchData['inputLabel']
        else:
            inputLabel = None
        if 'attentionFlag' in batchData.keys():
            attentionFlag = batchData['attentionFlag']
        else:
            attentionFlag = False

        inputData = inputData.float()
        if self.cudaFlag:
            inputData = inputData.cuda()
            inputSeqLen = inputSeqLen.cuda()
            if inputLabel is not None: inputLabel = inputLabel.cuda()

        rnnOutput, _ = self.rnnLayer(input=inputData, hx=None)
        attentionResult, attentionHotMap = self.ApplyAttention(
            dataInput=rnnOutput, attentionName=self.attentionName, inputSeqLen=inputSeqLen, hiddenNoduleNumbers=256)
        predict = self.predict(input=attentionResult)

        if attentionFlag: return attentionHotMap
        if inputLabel is not None: return self.lossFunction(input=predict, target=inputLabel)
        return predict

    def cudaTreatment(self):
        pass


class BLSTMwAttention_AttentionTransform(BLSTMwAttention):
    def __init__(self, attentionName, attentionScope, featuresNumber, classNumber, cudaFlag, transformWeight):
        self.transformWeight = transformWeight
        super(BLSTMwAttention_AttentionTransform, self).__init__(
            attentionName=attentionName, attentionScope=attentionScope, featuresNumber=featuresNumber,
            classNumber=classNumber, cudaFlag=cudaFlag)
        self.transformLossFunction = torch.nn.SmoothL1Loss()

    def __ZeroPadding(self, batchData, maxLen):
        padResult = []
        for sample in batchData:
            if len(sample) < maxLen: sample = numpy.concatenate([sample, numpy.zeros(maxLen - len(sample))])
            if len(sample) > maxLen: sample = sample[0:maxLen]
            padResult.append(sample)
        return numpy.array(padResult)

    def forward(self, batchData):
        inputData = batchData['inputData']
        inputSeqLen = batchData['inputSeqLen']

        if 'inputLabel' in batchData.keys():
            inputLabel = batchData['inputLabel']
        else:
            inputLabel = None
        if 'attentionFlag' in batchData.keys():
            attentionFlag = batchData['attentionFlag']
        else:
            attentionFlag = False

        if 'inputAttentionMap' in batchData.keys() and 'inputAttentionSeq' in batchData.keys():
            inputAttentionMap = batchData['inputAttentionMap'].detach().cpu().numpy()
            inputAttentionSeq = batchData['inputAttentionSeq'].numpy()

            shrinkedAttentionMaps = []
            for indexX in range(numpy.shape(inputAttentionMap)[0]):
                realAttentionMap = inputAttentionMap[indexX][0:inputAttentionSeq[indexX]]
                expandedAttentionMap = numpy.repeat(realAttentionMap, 1000)
                shrinkedAttentionMap = \
                    expandedAttentionMap[0::int(inputAttentionSeq[indexX] * 1000 / inputSeqLen[indexX])][
                    0:inputSeqLen[indexX]]
                shrinkedAttentionMaps.append(shrinkedAttentionMap)

        if self.cudaFlag:
            inputData = inputData.cuda()
            inputSeqLen = inputSeqLen.cuda()
            if inputLabel is not None: inputLabel = inputLabel.cuda()

        inputData = inputData.float()
        rnnOutput, _ = self.rnnLayer(input=inputData, hx=None)
        attentionResult, attentionHotMap = self.ApplyAttention(
            dataInput=rnnOutput, attentionName=self.attentionName, inputSeqLen=inputSeqLen, hiddenNoduleNumbers=256)
        predict = self.predict(input=attentionResult)

        #########################################################

        if attentionFlag: return attentionHotMap
        if inputLabel is not None and 'inputAttentionMap' in batchData.keys() and 'inputAttentionSeq' in batchData.keys():
            sourceMap = self.__ZeroPadding(batchData=shrinkedAttentionMaps, maxLen=attentionHotMap.size()[1])
            sourceMap = sourceMap[:, :, numpy.newaxis]
            sourceMap = torch.FloatTensor(sourceMap)
            if self.cudaFlag: sourceMap = sourceMap.cuda()
            return self.lossFunction(input=predict, target=inputLabel) + \
                   self.transformWeight * self.transformLossFunction(input=attentionHotMap, target=sourceMap)
        return predict
