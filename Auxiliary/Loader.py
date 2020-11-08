import os
import numpy
import torch
import torch.utils.data as torch_utils_data


class Collate_IEMOCAP_Base:
    def DataTensor1D(self, dataInput, maxLen):
        return numpy.concatenate([dataInput, torch.zeros([maxLen - numpy.shape(dataInput)[0]])])

    def DataTensor2D(self, dataInput, maxLen):
        return numpy.concatenate([dataInput, torch.zeros([maxLen - len(dataInput), numpy.shape(dataInput)[1]],
                                                         dtype=torch.float)], axis=0)

    def DataTensor3D(self, dataInput, maxLen):
        return numpy.concatenate([dataInput, torch.zeros(
            [numpy.shape(dataInput)[0], maxLen - numpy.shape(dataInput)[1], numpy.shape(dataInput)[2]],
            dtype=torch.float)], axis=1)


class Collate_IEMOCAP_SpectrumWlabel(Collate_IEMOCAP_Base):
    def __init__(self):
        super(Collate_IEMOCAP_SpectrumWlabel, self).__init__()

    def __call__(self, batch):
        xs = [v[0] for v in batch]
        ys = torch.LongTensor([v[1] for v in batch])

        xSeqLen = torch.LongTensor([v for v in map(len, xs)])
        xMaxLen = max([len(v) for v in xs])
        xs = numpy.array([self.DataTensor2D(dataInput=v, maxLen=xMaxLen) for v in xs], dtype=float)
        xs = torch.FloatTensor(xs)

        return xs, xSeqLen, ys


class Collate_IEMOCAP_BothAudioVideo(Collate_IEMOCAP_Base):
    def __init__(self):
        super(Collate_IEMOCAP_BothAudioVideo, self).__init__()

    def __call__(self, batch):
        xs = [v[0] for v in batch]
        ys = [v[1] for v in batch]
        zs = torch.LongTensor([v[2] for v in batch])

        xSeqLen = torch.LongTensor([v for v in map(len, xs)])
        xMaxLen = max([len(v) for v in xs])
        xs = numpy.array([self.DataTensor2D(dataInput=v, maxLen=xMaxLen) for v in xs], dtype=float)
        xs = torch.FloatTensor(xs)

        ySeqLen = torch.LongTensor([v for v in map(len, ys)])
        yMaxLen = max([len(v) for v in ys])
        ys = numpy.array([self.DataTensor2D(dataInput=v, maxLen=yMaxLen) for v in ys], dtype=float)
        ys = torch.FloatTensor(ys)

        return xs, xSeqLen, ys, ySeqLen, zs


class Dataset_IEMOCAP(torch_utils_data.Dataset):
    def __init__(self, data, label):
        self.data, self.label = data, label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.label[index]


class Dataset_IEMOCAP_BothAudioVideo(torch_utils_data.Dataset):
    def __init__(self, audioData, videoData, label):
        self.audioData, self.videoData, self.label = audioData, videoData, label

    def __len__(self):
        return len(self.audioData)

    def __getitem__(self, index):
        return self.audioData[index], self.videoData[index], self.label[index]


def LoadSpectrumWlabel(appointMedia, includePart=['improve', 'script'], appointGender=None, appointSession=None,
                       batchSize=16, shuffleFlag=True):
    loadPath = 'D:/PythonProjects_Data/IEMOCAP/DataSource_%s/' % appointMedia
    trainData, trainLabel, testData, testLabel = [], [], [], []

    for part in includePart:
        for gender in ['Female', 'Male']:
            for session in range(1, 6):
                currentData = numpy.load(
                    file=os.path.join(loadPath, '%s-%s-Session%d-Data.npy' % (part, gender, session)),
                    allow_pickle=True)
                currentLabel = numpy.load(
                    file=os.path.join(loadPath, '%s-%s-Session%d-Label.npy' % (part, gender, session)),
                    allow_pickle=True)

                if appointGender is not None and gender == appointGender and session == appointSession:
                    testData.extend(currentData)
                    testLabel.extend(currentLabel)
                else:
                    trainData.extend(currentData)
                    trainLabel.extend(currentLabel)
    print(numpy.shape(trainData), numpy.shape(trainLabel), numpy.sum(trainLabel, axis=0), numpy.shape(testData),
          numpy.shape(testLabel), numpy.sum(testLabel, axis=0))

    trainLabel = numpy.argmax(trainLabel, axis=1)
    trainDataset = Dataset_IEMOCAP(data=trainData, label=trainLabel)
    if len(testData) != 0:
        testLabel = numpy.argmax(testLabel, axis=1)
        testDataset = Dataset_IEMOCAP(data=testData, label=testLabel)

    if len(testData) != 0:
        return torch_utils_data.DataLoader(dataset=trainDataset, batch_size=batchSize, shuffle=shuffleFlag,
                                           collate_fn=Collate_IEMOCAP_SpectrumWlabel()), \
               torch_utils_data.DataLoader(dataset=testDataset, batch_size=batchSize, shuffle=False,
                                           collate_fn=Collate_IEMOCAP_SpectrumWlabel())
    else:
        return torch_utils_data.DataLoader(dataset=trainDataset, batch_size=batchSize, shuffle=shuffleFlag,
                                           collate_fn=Collate_IEMOCAP_SpectrumWlabel()), None


def LoadSpectrumWAudioVideo(includePart=['improve', 'script'], appointGender=None, appointSession=None,
                            batchSize=16, shuffleFlag=True):
    loadPath = 'D:/PythonProjects_Data/IEMOCAP/DataSource_Both/'
    trainAudioData, trainVideoData, trainLabel, testAudioData, testVideoData, testLabel = [], [], [], [], [], []

    for part in includePart:
        for gender in ['Female', 'Male']:
            for session in range(1, 6):
                currentAudioData = numpy.load(
                    file=os.path.join(loadPath, '%s-%s-Session%d-Audio.npy' % (part, gender, session)),
                    allow_pickle=True)
                currentVideoData = numpy.load(
                    file=os.path.join(loadPath, '%s-%s-Session%d-Video.npy' % (part, gender, session)),
                    allow_pickle=True)
                currentLabel = numpy.load(
                    file=os.path.join(loadPath, '%s-%s-Session%d-Label.npy' % (part, gender, session)),
                    allow_pickle=True)

                if appointGender is not None and gender == appointGender and session == appointSession:
                    testAudioData.extend(currentAudioData)
                    testVideoData.extend(currentVideoData)
                    testLabel.extend(currentLabel)
                else:
                    trainAudioData.extend(currentAudioData)
                    trainVideoData.extend(currentVideoData)
                    trainLabel.extend(currentLabel)
    print(numpy.shape(trainAudioData), numpy.shape(trainVideoData), numpy.shape(trainLabel),
          numpy.sum(trainLabel, axis=0), numpy.shape(testAudioData), numpy.shape(testVideoData), numpy.shape(testLabel),
          numpy.sum(testLabel, axis=0))

    trainLabel = numpy.argmax(trainLabel, axis=1)
    trainDataset = Dataset_IEMOCAP_BothAudioVideo(audioData=trainAudioData, videoData=trainVideoData, label=trainLabel)
    if len(testAudioData) != 0:
        testLabel = numpy.argmax(testLabel, axis=1)
        testDataset = Dataset_IEMOCAP_BothAudioVideo(audioData=testAudioData, videoData=testVideoData, label=testLabel)

    if len(testAudioData) != 0:
        return torch_utils_data.DataLoader(dataset=trainDataset, batch_size=batchSize, shuffle=shuffleFlag,
                                           collate_fn=Collate_IEMOCAP_BothAudioVideo()), \
               torch_utils_data.DataLoader(dataset=testDataset, batch_size=batchSize, shuffle=False,
                                           collate_fn=Collate_IEMOCAP_BothAudioVideo())
    else:
        return torch_utils_data.DataLoader(dataset=trainDataset, batch_size=batchSize, shuffle=shuffleFlag,
                                           collate_fn=Collate_IEMOCAP_BothAudioVideo()), None


if __name__ == '__main__':
    trainDataset, testDataset = LoadSpectrumWAudioVideo()
    for batchIndex, [batchAudioData, batchAudioSeq, batchVideoData, batchVideoSeq, batchLabel] in enumerate(
            trainDataset):
        print(numpy.shape(batchAudioData), numpy.shape(batchVideoData))
        # exit()
    #     # print(numpy.shape(batchData), numpy.shape(batchDataSeqLen), numpy.shape(batchLabel),
    #     #       numpy.shape(batchLabelSeqLen))
    #     # exit()
