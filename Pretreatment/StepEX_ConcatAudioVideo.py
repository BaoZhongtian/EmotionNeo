import os
import numpy

if __name__ == '__main__':
    audioPath = 'D:/PythonProjects_Data/IEMOCAP_Text&Audio/Step4_Normalization/'
    videoPath = 'D:/PythonProjects_DataNeo/IEMOCAP_VideoTreatment/Step5_Normalized_AllConcat/'
    savePath = 'D:/PythonProjects_DataNeo/VideoTreatment/DataSource/VASources/'
    if not os.path.exists(savePath): os.makedirs(savePath)

    emotionDictionary = {'ang': 0, 'exc': 1, 'hap': 1, 'neu': 2, 'sad': 3}
    for partName in os.listdir(videoPath):
        for genderName in os.listdir(os.path.join(videoPath, partName)):
            for sessionName in os.listdir(os.path.join(videoPath, partName, genderName)):
                partAudio, partVideo, partLabel = [], [], []
                for emotionName in os.listdir(os.path.join(videoPath, partName, genderName, sessionName)):
                    emotionLabel = numpy.zeros(4)
                    emotionLabel[emotionDictionary[emotionName]] = 1

                    for fileName in os.listdir(os.path.join(videoPath, partName, genderName, sessionName, emotionName)):
                        currentAudio = numpy.genfromtxt(
                            fname=os.path.join(audioPath, partName, genderName, sessionName, emotionName, fileName),
                            dtype=int, delimiter=',')
                        currentVideo = numpy.genfromtxt(
                            fname=os.path.join(videoPath, partName, genderName, sessionName, emotionName, fileName),
                            dtype=int, delimiter=',')

                        partAudio.append(currentAudio)
                        partVideo.append(currentVideo)
                        partLabel.append(emotionLabel)

                print(partName, genderName, sessionName, numpy.shape(partVideo), numpy.shape(partAudio),
                      numpy.shape(partLabel))

                numpy.save(file=os.path.join(savePath, '%s-%s-%s-Label.npy' % (partName, genderName, sessionName)),
                           arr=partLabel)
                numpy.save(file=os.path.join(savePath, '%s-%s-%s-Video.npy' % (partName, genderName, sessionName)),
                           arr=partVideo)
                numpy.save(file=os.path.join(savePath, '%s-%s-%s-Audio.npy' % (partName, genderName, sessionName)),
                           arr=partAudio)
