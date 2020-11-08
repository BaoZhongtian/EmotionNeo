import torch
import numpy


class AttentionBase(torch.nn.Module):
    def __init__(self, attentionName, attentionScope, featuresNumber, cudaFlag):
        self.attentionName, self.attentionScope, self.cudaFlag = attentionName, attentionScope, cudaFlag
        super(AttentionBase, self).__init__()
        if attentionName == 'StandardAttention':
            self.attentionWeightLayer = torch.nn.Linear(in_features=featuresNumber, out_features=1, bias=True)
        if attentionName == 'LocalAttention':
            self.attentionWeightLayer = torch.nn.Linear(in_features=featuresNumber * attentionScope, out_features=1,
                                                        bias=True)
        if self.attentionName == 'ComponentAttention':
            self.attentionWeightLayer = torch.nn.Conv2d(
                in_channels=1, out_channels=featuresNumber, kernel_size=[attentionScope, featuresNumber], stride=[1, 1],
                padding_mode='VALID')
        if self.attentionName == 'MonotonicAttention':
            self.sumKernel = torch.ones(size=[1, 1, self.attentionScope])
            self.attentionWeightNumeratorLayer = torch.nn.Linear(in_features=featuresNumber, out_features=1, bias=True)
            self.attentionWeightDenominatorLayer = torch.nn.Linear(in_features=featuresNumber, out_features=1,
                                                                   bias=True)
        if self.attentionName == 'SelfAttention':
            self.attentionKeyWeightLayer = torch.nn.Linear(in_features=featuresNumber, out_features=64, bias=True)
            self.attentionQueryWeightLayer = torch.nn.Linear(in_features=featuresNumber, out_features=64, bias=True)
            self.attentionValueWeightLayer = torch.nn.Linear(in_features=featuresNumber, out_features=featuresNumber,
                                                             bias=True)
            self.attentionWeightLayer = torch.nn.Linear(in_features=featuresNumber, out_features=1, bias=True)

    def ApplyAttention(self, dataInput, attentionName, inputSeqLen, hiddenNoduleNumbers):
        if attentionName == 'StandardAttention':
            return self.StandardAttention(
                dataInput=dataInput, seqInput=inputSeqLen, hiddenNoduleNumbers=hiddenNoduleNumbers)
        if attentionName == 'LocalAttention':
            return self.LocalAttention(
                dataInput=dataInput, seqInput=inputSeqLen, hiddenNoduleNumbers=hiddenNoduleNumbers)
        if attentionName == 'ComponentAttention':
            return self.ComponentAttention(
                dataInput=dataInput, seqInput=inputSeqLen, hiddenNoduleNumbers=hiddenNoduleNumbers)
        if attentionName == 'MonotonicAttention':
            return self.MonotonicAttention(
                dataInput=dataInput, seqInput=inputSeqLen, hiddenNoduleNumbers=hiddenNoduleNumbers)
        if attentionName == 'QuantumAttention':
            return self.QuantumAttention(
                dataInput=dataInput, seqInput=inputSeqLen, hiddenNoduleNumbers=hiddenNoduleNumbers)
        if attentionName == 'SelfAttention':
            return self.SelfAttention(
                dataInput=dataInput, seqInput=inputSeqLen, hiddenNoduleNumbers=hiddenNoduleNumbers)

    def AttentionMask(self, seqInput):
        returnTensor = torch.cat(
            [torch.cat([torch.ones(v), torch.ones(torch.max(seqInput) - v) * -1]).view([1, -1]) for v in seqInput])
        if self.cudaFlag:
            return returnTensor.cuda() * 9999
        else:
            return returnTensor * 9999

    def StandardAttention(self, dataInput, seqInput, hiddenNoduleNumbers):
        attentionOriginWeight = self.attentionWeightLayer(input=dataInput.reshape([-1, hiddenNoduleNumbers]))
        attentionOriginWeight = attentionOriginWeight.view([dataInput.size()[0], dataInput.size()[1]])

        if seqInput is not None:
            attentionMaskWeight = attentionOriginWeight.min(self.AttentionMask(seqInput=seqInput))
        else:
            attentionMaskWeight = attentionOriginWeight

        attentionWeight = torch.nn.functional.softmax(attentionMaskWeight, dim=-1).view([len(dataInput), -1, 1])
        attentionSupplementWeight = attentionWeight.repeat([1, 1, hiddenNoduleNumbers])
        attentionSeparateResult = torch.mul(dataInput, attentionSupplementWeight)
        attentionResult = attentionSeparateResult.sum(dim=1)
        return attentionResult, attentionWeight

    def LocalAttention(self, dataInput, seqInput, hiddenNoduleNumbers):
        dataInputPaddingPart = torch.zeros(size=[dataInput.size()[0], self.attentionScope, dataInput.size()[2]])
        if self.cudaFlag:
            dataInputPaddingPart = dataInputPaddingPart.cuda()
        dataInputSupplement = torch.cat([dataInput, dataInputPaddingPart], dim=1)
        dataInputExtension = torch.cat(
            [dataInputSupplement[:, v:dataInput.size()[1] + v, :] for v in range(self.attentionScope)], dim=-1)
        attentionOriginWeight = self.attentionWeightLayer(
            input=dataInputExtension.view([-1, hiddenNoduleNumbers * self.attentionScope])).view(
            [dataInput.size()[0], -1])
        #########################################################

        if seqInput is not None:
            attentionMaskWeight = attentionOriginWeight.min(self.AttentionMask(seqInput=seqInput))
        else:
            attentionMaskWeight = attentionOriginWeight

        attentionWeight = torch.nn.functional.softmax(attentionMaskWeight, dim=-1).view([len(dataInput), -1, 1])
        attentionSupplementWeight = attentionWeight.repeat([1, 1, hiddenNoduleNumbers])
        attentionSeparateResult = torch.mul(dataInput, attentionSupplementWeight)
        attentionResult = attentionSeparateResult.sum(dim=1)
        return attentionResult, attentionWeight

    def ComponentAttention(self, dataInput, seqInput, hiddenNoduleNumbers):
        dataInputPaddingPart = torch.zeros(size=[dataInput.size()[0], self.attentionScope - 1, dataInput.size()[2]])
        if self.cudaFlag:
            dataInputPaddingPart = dataInputPaddingPart.cuda()
        dataInputSupplement = torch.cat([dataInput, dataInputPaddingPart], dim=1)
        dataInputSupplement = dataInputSupplement.unsqueeze(1)
        attentionOriginWeight = self.attentionWeightLayer(input=dataInputSupplement).squeeze()
        if len(attentionOriginWeight.size()) == 2: attentionOriginWeight = attentionOriginWeight.unsqueeze(0)
        attentionOriginWeight = attentionOriginWeight.permute(0, 2, 1)

        if seqInput is not None:
            attentionMask = self.AttentionMask(seqInput=seqInput).unsqueeze(-1).repeat([1, 1, hiddenNoduleNumbers])
            attentionMaskWeight = attentionOriginWeight.min(attentionMask)
        else:
            attentionMaskWeight = attentionOriginWeight

        attentionWeight = torch.nn.functional.softmax(attentionMaskWeight, dim=1)
        attentionSeparateResult = torch.mul(dataInput, attentionWeight)
        attentionResult = attentionSeparateResult.sum(dim=1)
        return attentionResult, attentionWeight

    def MonotonicAttention(self, dataInput, seqInput, hiddenNoduleNumbers):
        attentionNumeratorWeight = self.attentionWeightNumeratorLayer(input=dataInput).tanh()
        attentionDenominatorRawWeight = self.attentionWeightDenominatorLayer(input=dataInput).exp()
        padDenominatorZero = torch.zeros(size=[attentionDenominatorRawWeight.size()[0], self.attentionScope - 1,
                                               attentionDenominatorRawWeight.size()[2]])
        if self.cudaFlag:
            padDenominatorZero = padDenominatorZero.cuda()
            self.sumKernel = self.sumKernel.float().cuda()

        attentionDenominatorSupplementWeight = torch.cat([padDenominatorZero, attentionDenominatorRawWeight], dim=1)

        attentionDenominatorWeight = torch.conv1d(input=attentionDenominatorSupplementWeight.permute(0, 2, 1),
                                                  weight=self.sumKernel, stride=1)
        attentionOriginWeight = torch.div(attentionNumeratorWeight.squeeze(), attentionDenominatorWeight.squeeze())

        #########################################################

        if seqInput is not None:
            attentionMaskWeight = attentionOriginWeight.min(self.AttentionMask(seqInput=seqInput))
        else:
            attentionMaskWeight = attentionOriginWeight
        attentionWeight = torch.nn.functional.softmax(attentionMaskWeight, dim=-1).view([len(dataInput), -1, 1])
        attentionSupplementWeight = attentionWeight.repeat([1, 1, hiddenNoduleNumbers])
        attentionSeparateResult = torch.mul(dataInput, attentionSupplementWeight)
        attentionResult = attentionSeparateResult.sum(dim=1)
        return attentionResult, attentionWeight

    def QuantumAttention(self, dataInput, seqInput, hiddenNoduleNumbers):
        initialState = dataInput[:, 0, :]
        initialStateRepeat = initialState.unsqueeze(1).repeat([1, dataInput.size()[1], 1])
        initialMultiplyResult = torch.mul(dataInput, initialStateRepeat)

        finalState = torch.cat([dataInput[v:v + 1, seqInput[v] - 1, :] for v in range(len(seqInput))], dim=0)
        finalStateRespeat = finalState.unsqueeze(1).repeat([1, dataInput.size()[1], 1])
        finalMultiplyResult = torch.mul(dataInput, finalStateRespeat)

        attentionResultMatric = torch.cat(
            [initialMultiplyResult[v].matmul(finalMultiplyResult[v].transpose(1, 0)).unsqueeze(0) for v in
             range(dataInput.size()[0])], dim=0)
        attentionEye = torch.eye(n=dataInput.size()[1]).unsqueeze(0).repeat([dataInput.size()[0], 1, 1])
        if self.cudaFlag: attentionEye = attentionEye.cuda()
        attentionOriginWeight = torch.mul(attentionResultMatric, attentionEye).sum(dim=1)

        if seqInput is not None:
            attentionMaskWeight = attentionOriginWeight.min(self.AttentionMask(seqInput=seqInput))
        else:
            attentionMaskWeight = attentionOriginWeight
        attentionWeight = torch.nn.functional.softmax(attentionMaskWeight, dim=-1).view([len(dataInput), -1, 1])
        attentionSupplementWeight = attentionWeight.repeat([1, 1, hiddenNoduleNumbers])
        attentionSeparateResult = torch.mul(dataInput, attentionSupplementWeight)
        attentionResult = attentionSeparateResult.sum(dim=1)
        return attentionResult, attentionWeight

    def SelfAttention(self, dataInput, seqInput, hiddenNoduleNumbers):
        attentionKeyWeight = self.attentionKeyWeightLayer(dataInput)
        attentionQueryWeight = self.attentionQueryWeightLayer(dataInput)
        attentionValueWeight = self.attentionValueWeightLayer(dataInput)
        attentionKQ = attentionKeyWeight.bmm(attentionQueryWeight.permute(0, 2, 1)) / 8
        attentionKQ = attentionKQ.softmax(dim=-1)
        attentionResult = attentionKQ.bmm(attentionValueWeight).sum(dim=1)
        return attentionResult, attentionKQ

        # print(numpy.shape(attentionKeyWeight), numpy.shape(attentionQueryWeight), numpy.shape(attentionValueWeight))
        # print(numpy.shape(attentionKQ), numpy.shape(attentionResult))
        # exit()
        # return self.ApplyAttention(dataInput=attentionResult, attentionName='StandardAttention', inputSeqLen=seqInput,
        #                            hiddenNoduleNumbers=hiddenNoduleNumbers)
        # print(numpy.shape(attentionResult))
        # exit()


class AttentionBase_Multi(torch.nn.Module):
    def __init__(self, attentionName, attentionScope, attentionParameter, featuresNumber, cudaFlag):
        self.attentionName, self.attentionScope, self.attentionParameter, self.cudaFlag = \
            attentionName, attentionScope, attentionParameter, cudaFlag
        super(AttentionBase_Multi, self).__init__()
        if len(attentionName) < 2 or len(attentionScope) < 2 or len(attentionParameter) < 2:
            raise RuntimeError('Please give MultiAttention')
        if not len(attentionName) == len(attentionScope) == len(attentionParameter):
            raise RuntimeError('Please Give Same Length')

        self.attentionWeightLayer, self.sumKernel, self.attentionWeightNumeratorLayer, self.attentionWeightDenominatorLayer = {}, {}, {}, {}
        self.attentionKeyWeightLayer, self.attentionQueryWeightLayer, self.attentionValueWeightLayer = {}, {}, {}
        for index in range(len(attentionName)):
            if attentionName[index] == 'StandardAttention':
                self.attentionWeightLayer[attentionParameter[index]] = \
                    torch.nn.Linear(in_features=featuresNumber, out_features=1)
            if attentionName[index] == 'LocalAttention':
                self.attentionWeightLayer[attentionParameter[index]] = \
                    torch.nn.Linear(in_features=featuresNumber * attentionScope[index], out_features=1)
            if attentionName[index] == 'ComponentAttention':
                self.attentionWeightLayer[attentionParameter[index]] = torch.nn.Conv2d(
                    in_channels=1, out_channels=featuresNumber, kernel_size=[attentionScope[index], featuresNumber],
                    stride=[1, 1])
            if self.attentionName[index] == 'MonotonicAttention':
                self.sumKernel[attentionParameter[index]] = torch.ones(size=[1, 1, self.attentionScope[index]])
                self.attentionWeightNumeratorLayer[attentionParameter[index]] = torch.nn.Linear(
                    in_features=featuresNumber, out_features=1)
                self.attentionWeightDenominatorLayer[attentionParameter[index]] = torch.nn.Linear(
                    in_features=featuresNumber, out_features=1)

            if self.attentionName[index] == 'SelfAttention':
                self.attentionKeyWeightLayer[attentionParameter[index]] = torch.nn.Linear(
                    in_features=featuresNumber, out_features=64, bias=True)
                self.attentionQueryWeightLayer[attentionParameter[index]] = torch.nn.Linear(
                    in_features=featuresNumber, out_features=64, bias=True)
                self.attentionValueWeightLayer[attentionParameter[index]] = torch.nn.Linear(
                    in_features=featuresNumber, out_features=featuresNumber, bias=True)

    def ApplyAttention(self, dataInput, attentionName, attentionParameter, inputSeqLen, hiddenNoduleNumbers):
        if attentionName == 'StandardAttention':
            return self.StandardAttention(
                dataInput=dataInput, seqInput=inputSeqLen, attentionParameter=attentionParameter,
                hiddenNoduleNumbers=hiddenNoduleNumbers)
        if attentionName == 'LocalAttention':
            return self.LocalAttention(
                dataInput=dataInput, seqInput=inputSeqLen, attentionParameter=attentionParameter,
                attentionScope=self.attentionScope[self.attentionParameter.sessionIndex(attentionParameter)],
                hiddenNoduleNumbers=hiddenNoduleNumbers)
        if attentionName == 'ComponentAttention':
            return self.ComponentAttention(
                dataInput=dataInput, seqInput=inputSeqLen, attentionParameter=attentionParameter,
                attentionScope=self.attentionScope[self.attentionParameter.sessionIndex(attentionParameter)],
                hiddenNoduleNumbers=hiddenNoduleNumbers)
        if attentionName == 'MonotonicAttention':
            return self.MonotonicAttention(
                dataInput=dataInput, seqInput=inputSeqLen, attentionParameter=attentionParameter,
                attentionScope=self.attentionScope[self.attentionParameter.sessionIndex(attentionParameter)],
                hiddenNoduleNumbers=hiddenNoduleNumbers)
        if attentionName == 'SelfAttention':
            return self.SelfAttention(
                dataInput=dataInput, seqInput=inputSeqLen, attentionParameter=attentionParameter,
                attentionScope=self.attentionScope[self.attentionParameter.sessionIndex(attentionParameter)],
                hiddenNoduleNumbers=hiddenNoduleNumbers)

    def AttentionMask(self, seqInput):
        returnTensor = torch.cat(
            [torch.cat([torch.ones(v), torch.ones(torch.max(seqInput) - v) * -1]).view([1, -1]) for v in seqInput])
        if self.cudaFlag:
            return returnTensor.cuda() * 9999
        else:
            return returnTensor * 9999

    def StandardAttention(self, attentionParameter, dataInput, seqInput, hiddenNoduleNumbers):
        attentionOriginWeight = self.attentionWeightLayer[attentionParameter](
            input=dataInput.reshape([-1, hiddenNoduleNumbers]))
        attentionOriginWeight = attentionOriginWeight.view([dataInput.size()[0], dataInput.size()[1]])

        if seqInput is not None:
            attentionMaskWeight = attentionOriginWeight.min(self.AttentionMask(seqInput=seqInput))
        else:
            attentionMaskWeight = attentionOriginWeight

        attentionWeight = torch.nn.functional.softmax(attentionMaskWeight, dim=-1).view([len(dataInput), -1, 1])
        attentionSupplementWeight = attentionWeight.repeat([1, 1, hiddenNoduleNumbers])
        attentionSeparateResult = torch.mul(dataInput, attentionSupplementWeight)
        attentionResult = attentionSeparateResult.sum(dim=1)
        return attentionResult, attentionWeight

    def LocalAttention(self, attentionParameter, attentionScope, dataInput, seqInput, hiddenNoduleNumbers):
        dataInputPaddingPart = torch.zeros(size=[dataInput.size()[0], attentionScope, dataInput.size()[2]])
        if self.cudaFlag:
            dataInputPaddingPart = dataInputPaddingPart.cuda()
        dataInputSupplement = torch.cat([dataInput, dataInputPaddingPart], dim=1)
        dataInputExtension = torch.cat(
            [dataInputSupplement[:, v:dataInput.size()[1] + v, :] for v in range(attentionScope)], dim=-1)
        attentionOriginWeight = self.attentionWeightLayer[attentionParameter](
            input=dataInputExtension.view([-1, hiddenNoduleNumbers * attentionScope])).view(
            [dataInput.size()[0], -1])
        #########################################################

        if seqInput is not None:
            attentionMaskWeight = attentionOriginWeight.min(self.AttentionMask(seqInput=seqInput))
        else:
            attentionMaskWeight = attentionOriginWeight

        attentionWeight = torch.nn.functional.softmax(attentionMaskWeight, dim=-1).view([len(dataInput), -1, 1])
        attentionSupplementWeight = attentionWeight.repeat([1, 1, hiddenNoduleNumbers])
        attentionSeparateResult = torch.mul(dataInput, attentionSupplementWeight)
        attentionResult = attentionSeparateResult.sum(dim=1)
        return attentionResult, attentionWeight

    def ComponentAttention(self, attentionParameter, attentionScope, dataInput, seqInput, hiddenNoduleNumbers):
        dataInputPaddingPart = torch.zeros(size=[dataInput.size()[0], attentionScope - 1, dataInput.size()[2]])
        if self.cudaFlag:
            dataInputPaddingPart = dataInputPaddingPart.cuda()
        dataInputSupplement = torch.cat([dataInput, dataInputPaddingPart], dim=1)
        dataInputSupplement = dataInputSupplement.unsqueeze(1)
        attentionOriginWeight = self.attentionWeightLayer[attentionParameter](input=dataInputSupplement).squeeze()
        if len(attentionOriginWeight.size()) == 2: attentionOriginWeight = attentionOriginWeight.unsqueeze(0)
        attentionOriginWeight = attentionOriginWeight.permute(0, 2, 1)

        if seqInput is not None:
            attentionMask = self.AttentionMask(seqInput=seqInput).unsqueeze(-1).repeat([1, 1, hiddenNoduleNumbers])
            attentionMaskWeight = attentionOriginWeight.min(attentionMask)
        else:
            attentionMaskWeight = attentionOriginWeight

        attentionWeight = torch.nn.functional.softmax(attentionMaskWeight, dim=1)
        attentionSeparateResult = torch.mul(dataInput, attentionWeight)
        attentionResult = attentionSeparateResult.sum(dim=1)
        return attentionResult, attentionWeight

    def MonotonicAttention(self, attentionParameter, attentionScope, dataInput, seqInput, hiddenNoduleNumbers):
        attentionNumeratorWeight = self.attentionWeightNumeratorLayer[attentionParameter](input=dataInput).tanh()
        attentionDenominatorRawWeight = self.attentionWeightDenominatorLayer[attentionParameter](input=dataInput).exp()
        padDenominatorZero = torch.zeros(size=[attentionDenominatorRawWeight.size()[0], attentionScope - 1,
                                               attentionDenominatorRawWeight.size()[2]])
        if self.cudaFlag: padDenominatorZero = padDenominatorZero.cuda()

        attentionDenominatorSupplementWeight = torch.cat([padDenominatorZero, attentionDenominatorRawWeight], dim=1)

        attentionDenominatorWeight = torch.conv1d(input=attentionDenominatorSupplementWeight.permute(0, 2, 1),
                                                  weight=self.sumKernel[attentionParameter], stride=1)
        attentionOriginWeight = torch.div(attentionNumeratorWeight.squeeze(), attentionDenominatorWeight.squeeze())

        #########################################################

        if seqInput is not None:
            attentionMaskWeight = attentionOriginWeight.min(self.AttentionMask(seqInput=seqInput))
        else:
            attentionMaskWeight = attentionOriginWeight
        attentionWeight = torch.nn.functional.softmax(attentionMaskWeight, dim=-1).view([len(dataInput), -1, 1])
        attentionSupplementWeight = attentionWeight.repeat([1, 1, hiddenNoduleNumbers])
        attentionSeparateResult = torch.mul(dataInput, attentionSupplementWeight)
        attentionResult = attentionSeparateResult.sum(dim=1)
        return attentionResult, attentionWeight

    def SelfAttention(self, attentionParameter, attentionScope, dataInput, seqInput, hiddenNoduleNumbers):
        attentionKeyWeight = self.attentionKeyWeightLayer[attentionParameter](dataInput)
        attentionQueryWeight = self.attentionQueryWeightLayer[attentionParameter](dataInput)
        attentionValueWeight = self.attentionValueWeightLayer[attentionParameter](dataInput)
        attentionKQ = attentionKeyWeight.bmm(attentionQueryWeight.permute(0, 2, 1)) / 8
        attentionKQ = attentionKQ.softmax(dim=-1)
        attentionResult = attentionKQ.bmm(attentionValueWeight).sum(dim=1)
        return attentionResult, attentionKQ

    def cudaTreatment(self):
        for sample in self.attentionWeightLayer: self.attentionWeightLayer[sample].cuda()
        for sample in self.sumKernel: self.sumKernel[sample] = self.sumKernel[sample].float().cuda()
        for sample in self.attentionWeightNumeratorLayer: self.attentionWeightNumeratorLayer[sample].cuda()
        for sample in self.attentionWeightDenominatorLayer: self.attentionWeightDenominatorLayer[sample].cuda()
        for sample in self.attentionKeyWeightLayer: self.attentionKeyWeightLayer[sample].cuda()
        for sample in self.attentionQueryWeightLayer: self.attentionQueryWeightLayer[sample].cuda()
        for sample in self.attentionValueWeightLayer: self.attentionValueWeightLayer[sample].cuda()
