__author__ = 'chenchuang'

import argparse
import time
import matplotlib.pyplot as plt
import csv
import os
from scipy import ndimage
from tqdm import *

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from torchvision.transforms import *
import torchnet

from datasets import *
from transforms import *

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dataset-dir", type=str, default='datasets/speech_commands/test', help='path of test dataset')
parser.add_argument("--batch-size", type=int, default=96, help='batch size')
parser.add_argument("--dataload-workers-nums", type=int, default=3, help='number of workers for dataloader')
parser.add_argument("--input", choices=['mel32','mel40'], default='mel32', help='input of NN')
parser.add_argument('--multi-crop', action='store_true', help='apply crop and average the results')
parser.add_argument('--generate-kaggle-submission', action='store_true', help='generate kaggle submission file')
parser.add_argument("--kaggle-dataset-dir", type=str, default='datasets/speech_commands/kaggle', help='path of kaggle test dataset')
parser.add_argument('--output', type=str, default='', help='save output to file for the kaggle competition, if empty the model name will be used')
#parser.add_argument('--prob-output', type=str, help='save probabilities to file', default='probabilities.json')
parser.add_argument("--model", type=str, default='best-acc.pth',  help='a pretrained neural network model')
args = parser.parse_args()

dataset_dir = args.dataset_dir
if args.generate_kaggle_submission:
    dataset_dir = args.kaggle_dataset_dir

print("loading model...")
model = torch.load(args.model)
model.float()

use_gpu = torch.cuda.is_available()
print('use_gpu', use_gpu)
if use_gpu:
    torch.backends.cudnn.benchmark = True
    model.cuda()
#
n_mels = 32
if args.input == 'mel40':
    n_mels = 40

feature_transform = Compose([ToMelSpectrogram(n_mels=n_mels), ToTensor('mel_spectrogram', 'input')])
transform = Compose([LoadAudio(), FixAudioLength(), feature_transform])
test_dataset = SpeechCommandsDataset(dataset_dir, transform, silence_percentage=0)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, sampler=None,
                            pin_memory=use_gpu, num_workers=args.dataload_workers_nums)

criterion = torch.nn.CrossEntropyLoss()

def multi_crop(inputs):
    b = 1
    size = inputs.size(3) - b * 2
    patches = [inputs[:, :, :, i*b:size+i*b] for i in range(3)]
    outputs = torch.stack(patches)
    outputs = outputs.view(-1, inputs.size(1), inputs.size(2), size)
    outputs = torch.nn.functional.pad(outputs, (b, b, 0, 0), mode='replicate')
    return torch.cat((inputs, outputs.data))

def test():
    model.eval()  # Set model to evaluate mode

    #running_loss = 0.0
    #it = 0
    correct = 0
    total = 0
    confusion_matrix = torchnet.meter.ConfusionMeter(len(CLASSES))
    predictions = {}
    probabilities = {}

    pbar = tqdm(test_dataloader, unit="audios", unit_scale=test_dataloader.batch_size)
    for batch in pbar:
        inputs = batch['input']
        inputs = torch.unsqueeze(inputs, 1)
        targets = batch['target']

        n = inputs.size(0)
        if args.multi_crop:
            inputs = multi_crop(inputs)
        with torch.no_grad():
            inputs = Variable(inputs)
        targets = Variable(targets, requires_grad=False)

        if use_gpu:
            inputs = inputs.cuda()
            targets = targets.cuda(non_blocking=True)

        # forward
        outputs = model(inputs)
        #loss = criterion(outputs, targets)
        outputs = torch.nn.functional.softmax(outputs, dim=1)
        if args.multi_crop:
            outputs = outputs.view(-1, n, outputs.size(1))
            outputs = torch.mean(outputs, dim=0)
            outputs = torch.nn.functional.softmax(outputs, dim=1)

        # statistics
        #it += 1
        #running_loss += loss.data[0]
        # pred = outputs.data.max(1, keepdim=True)[1]
        pred = outputs.data.max(1, keepdim=False)[1]
        correct += pred.eq(targets.data.view_as(pred)).sum()
        total += targets.size(0)
        targets = targets.view(-1)
        confusion_matrix.add(pred, targets.data)

        filenames = batch['path']
        for j in range(len(pred)):
            fn = filenames[j]
            # predictions[fn] = pred[j][0]
            predictions[fn] = pred[j].item()
            probabilities[fn] = outputs.data[j].tolist()

    accuracy = correct/total
    #epoch_loss = running_loss / it
    print("accuracy: %f%%" % (100*accuracy))
    print("confusion matrix:")
    import numpy as np
    np.set_printoptions(threshold=np.inf)

    print(confusion_matrix.value())
    confusion_matrix_value = confusion_matrix.value()
    # 以下为新添加的部分，用来保存混淆矩阵为图像
    plt.figure(figsize=(10,10))  # 新建一个图像
    # 用imshow显示混淆矩阵
    plt.imshow(confusion_matrix_value, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    # 在这里添加代码
    fmt = 'd'  # 或者你想要的任何格式
    thresh = confusion_matrix_value.max() / 2.
    for i in range(confusion_matrix_value.shape[0]):
        for j in range(confusion_matrix_value.shape[1]):
            plt.text(j, i, format(confusion_matrix_value[i, j], fmt),
                     horizontalalignment="center",
                     fontsize=8,
                     color="white" if confusion_matrix_value[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("confusion_matrix.png")  # 保存为png图像
    plt.close()  # 关闭图像

    return probabilities, predictions

print("testing...")
probabilities, predictions = test()
if args.generate_kaggle_submission:
    output_file_name = "%s" % os.path.splitext(os.path.basename(args.model))[0]
    if args.multi_crop:
        output_file_name = "%s-crop" % output_file_name
    output_file_name = "%s.csv" % output_file_name
    if args.output:
        output_file_name = args.output
    print("generating kaggle submission file '%s'..." % output_file_name)
    with open(output_file_name, 'w') as outfile:
        fieldnames = ['fname', 'label']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for fname, pred in predictions.items():
            writer.writerow({'fname': os.path.basename(fname), 'label': test_dataset.classes[pred]})



def predict(audio_path):
    model.eval()  # Set model to evaluate mode
    CLASSES = 'unknown, zero, yes, wow, visual, up, two, tree, three, stop, six, sheila, seven, ' \
              'right, one, on, off, no, nine, marvin, left, learn, house, happy, go, four, ' \
              'forward, follow, five, eight, down, dog, cat, bird, bed, backward'.split(', ')
    result = []
    targets = []
    # 加载和预处理音频，同样需要根据你的音频数据预处理步骤进行=
    # 这部分代码需要你根据实际情况进行修改
    feature_transform = Compose([ToMelSpectrogram(n_mels=n_mels), ToTensor('mel_spectrogram', 'input')])
    transform = Compose([LoadAudio(), FixAudioLength(), feature_transform])
    audio_dataset = SpeechCommandsDataset(audio_path, transform, silence_percentage=0)
    audio_dataloader = DataLoader(audio_dataset, batch_size=1, sampler=None,
                                 pin_memory=use_gpu, num_workers=args.dataload_workers_nums)
    pbar = tqdm(audio_dataloader, unit="audios", unit_scale=audio_dataloader.batch_size)
    for batch in pbar:
        inputs = batch['input']
        target = batch['target']  # 获取正确的标签
        targets.append(target.item())
        # print("正确标签为：",targets)
        inputs = torch.unsqueeze(inputs, 1)
        if args.multi_crop:
            inputs = multi_crop(inputs)
        with torch.no_grad():
            inputs = Variable(inputs)
        if use_gpu:
            inputs = inputs.cuda()

        # forward
        outputs = model(inputs)
        outputs = torch.nn.functional.softmax(outputs, dim=1)
        if args.multi_crop:
            outputs = outputs.view(-1, n, outputs.size(1))
            outputs = torch.mean(outputs, dim=0)
            outputs = torch.nn.functional.softmax(outputs, dim=1)

        pred = outputs.data.max(1, keepdim=False)[1]
        result.append(pred.item())
        # 判断预测是否正确
    return result,targets

        # probabilities = outputs.data.tolist()
    #
    # print("predicted class:", pred.item())
    # print("class probabilities:", probabilities)

    # return pred.item()

print("predicting...")
audio_path = '/home/cc/Code/pytorch-speech-commands-master/datasets/speech_commands/my_recording'  # 替换为你实际的音频路径
result,targets=predict(audio_path)
print(result)
print(targets)