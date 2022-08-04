import os
import csv
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from scipy import signal
import random
import json
import xml.etree.ElementTree as ET
from audio_io import load_audio_av, open_audio_av
import torch

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def load_image(path):
    return Image.open(path).convert('RGB')


def load_spectrogram(path, dur=3.):
    # Load audio
    audio_ctr = open_audio_av(path)
    audio_dur = audio_ctr.streams.audio[0].duration * audio_ctr.streams.audio[0].time_base
    audio_ss = max(float(audio_dur)/2 - dur/2, 0)
    audio, samplerate = load_audio_av(container=audio_ctr, start_time=audio_ss, duration=dur)

    # To Mono
    audio = np.clip(audio, -1., 1.).mean(0)

    # Repeat if audio is too short
    if audio.shape[0] < samplerate * dur:
        n = int(samplerate * dur / audio.shape[0]) + 1
        audio = np.tile(audio, n)
    audio = audio[:int(samplerate * dur)]

    frequencies, times, spectrogram = signal.spectrogram(audio, samplerate, nperseg=512, noverlap=274)
    spectrogram = np.log(spectrogram + 1e-7)
    return spectrogram

def load_all_bboxes(annotation_dir, format='flickr'):
    gt_bboxes = {}
    if format == 'flickr':
        anno_files = os.listdir(annotation_dir)
        for filename in anno_files:
            file = filename.split('.')[0]
            gt = ET.parse(f"{annotation_dir}/{filename}").getroot()
            bboxes = []
            for child in gt:
                for childs in child:
                    bbox = []
                    if childs.tag == 'bbox':
                        for index, ch in enumerate(childs):
                            if index == 0:
                                continue
                            bbox.append(int(224 * int(ch.text)/256))
                    bboxes.append(bbox)
            gt_bboxes[file] = bboxes

    elif format == 'vggss':
        with open('metadata/vggss.json') as json_file:
            annotations = json.load(json_file)
        for annotation in annotations:
            bboxes = [(np.clip(np.array(bbox), 0, 1) * 224).astype(int) for bbox in annotation['bbox']]
            gt_bboxes[annotation['file']] = bboxes

    return gt_bboxes


def bbox2gtmap(bboxes, format='flickr'):
    gt_map = np.zeros([224, 224])
    for xmin, ymin, xmax, ymax in bboxes:
        temp = np.zeros([224, 224])
        temp[ymin:ymax, xmin:xmax] = 1
        gt_map += temp

    if format == 'flickr':
        # Annotation consensus
        gt_map = gt_map / 2
        gt_map[gt_map > 1] = 1

    elif format == 'vggss':
        # Single annotation
        gt_map[gt_map > 0] = 1

    return gt_map


class AudioVisualDataset(Dataset):
    def __init__(self, image_files, audio_files, image_path, audio_path, mode='train', sup_image_path=None, sup_audio_path=None, audio_dur=3., image_transform=None, audio_transform=None, all_bboxes=None, bbox_format='flickr'):
        super().__init__()
        self.audio_path = audio_path
        self.image_path = image_path

        self.mode = mode
        self.sup_audio_path = sup_audio_path
        self.sup_image_path = sup_image_path

        self.audio_dur = audio_dur

        self.audio_files = audio_files
        self.image_files = image_files
        self.all_bboxes = all_bboxes
        self.bbox_format = bbox_format

        self.image_transform = image_transform
        self.audio_transform = audio_transform

    def getitem(self, idx):

        image_path = self.image_path
        audio_path = self.audio_path

        anno = {}
        if self.all_bboxes is not None:
            bboxes = self.all_bboxes[idx]
            bb = -torch.ones((10, 4)).long()
            if len(bboxes) > 0:
                bb[:len(bboxes)] = torch.from_numpy(np.array(bboxes))
                anno['bboxes'] = bb
                anno['gt_map'] = bbox2gtmap(bboxes, self.bbox_format)
                anno['gt_mask'] = 1             # 1 for samples w. gt_map
                if self.mode == 'train':
                    image_path = self.sup_image_path
                    audio_path = self.sup_audio_path
            else:
                anno['bboxes'] = bb
                anno['gt_map'] = np.zeros([224, 224])
                anno['gt_mask'] = 0             # 0 for samples w/o. gt_map

        file = self.image_files[idx]
        file_id = file.split('.')[0]

        # Image
        img_fn = image_path + self.image_files[idx]
        frame = self.image_transform(load_image(img_fn))

        # Audio
        audio_fn = audio_path + self.audio_files[idx]
        spectrogram = self.audio_transform(load_spectrogram(audio_fn))

        return frame, spectrogram, anno, file_id

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        try:
            return self.getitem(idx)
        except Exception:
            return self.getitem(random.sample(range(len(self)), 1)[0])


def get_train_dataset(args):
    audio_path = f"{args.train_data_path}/audio/"
    image_path = f"{args.train_data_path}/frames/"

    # List directory
    audio_files = {fn.split('.wav')[0] for fn in os.listdir(audio_path) if fn.endswith('.wav')}
    image_files = {fn.split('.jpg')[0] for fn in os.listdir(image_path) if fn.endswith('.jpg')}
    avail_files = audio_files.intersection(image_files)
    print(f"{len(avail_files)} available files")

    # Subsample if specified
    if args.trainset.lower() in {'vggss', 'flickr'}:
        pass    # use full dataset
    else:
        subset = set(open(f"metadata/{args.trainset}.txt").read().splitlines())
        avail_files = avail_files.intersection(subset)
        print(f"{len(avail_files)} valid subset files")
    avail_files = sorted(list(avail_files))
    audio_files = sorted([dt+'.wav' for dt in avail_files])
    image_files = sorted([dt+'.jpg' for dt in avail_files])
    all_bboxes = [[] for _ in range(len(image_files))]

    # NOTE: load 4750 training files with grouth truth 
    if args.use_supervised_data:
        sup_audio_path = f"{args.sup_train_data_path}/audio/"
        sup_image_path = f"{args.sup_train_data_path}/frames/"

        #  Retrieve list of audio and video files
        sup_train_txt = 'metadata/flickr_sup_train.txt'
        supset = set(open(sup_train_txt).read().splitlines())

        # Intersect with available files
        sup_audio_files = {fn.split('.wav')[0] for fn in os.listdir(sup_audio_path)}
        sup_image_files = {fn.split('.jpg')[0] for fn in os.listdir(sup_image_path)}
        sup_avail_files = sup_audio_files.intersection(sup_image_files)
        supset = supset.intersection(sup_avail_files)

        supset = sorted(list(supset))
        print(f"{len(supset)} supervised training subset files")
        sup_image_files = [dt+'.jpg' for dt in supset]
        sup_audio_files = [dt+'.wav' for dt in supset]

        # Bounding boxes
        sup_bbox_format = 'flickr'
        sup_all_bboxes = load_all_bboxes(args.sup_train_gt_path, format=sup_bbox_format)
        sup_all_bboxes = [sup_all_bboxes[fn.split('.jpg')[0]] for fn in sup_image_files]
        
        # extend to the original set
        audio_files.extend(sup_audio_files)
        image_files.extend(sup_image_files)
        all_bboxes.extend(sup_all_bboxes)

        idx = list(range(len(image_files)))
        random.shuffle(idx)
        image_files = [image_files[i] for i in idx]
        audio_files = [audio_files[i] for i in idx]
        all_bboxes = [all_bboxes[i] for i in idx]
    
    else:
        sup_bbox_format = None
        sup_audio_path = None
        sup_image_path = None

    # Transforms
    image_transform = transforms.Compose([
        transforms.Resize(int(224 * 1.1), Image.BICUBIC),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    audio_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.0], std=[12.0])])

    return AudioVisualDataset(
        mode='train',
        image_files=image_files,
        audio_files=audio_files,
        all_bboxes=all_bboxes,
        bbox_format=sup_bbox_format,
        sup_audio_path=sup_audio_path,
        sup_image_path=sup_image_path,
        image_path=image_path,
        audio_path=audio_path,
        audio_dur=3.,
        image_transform=image_transform,
        audio_transform=audio_transform
    )


def get_test_dataset(args):
    audio_path = args.test_data_path + 'audio/'
    image_path = args.test_data_path + 'frames/'

    if args.testset in ['flickr', 'flickr_plus_silent']:
        testcsv = 'metadata/flickr_test.csv'
    elif args.testset in ['vggss', 'vggss_plus_silent']:
        testcsv = 'metadata/vggss_test.csv'
    elif args.testset == 'vggss_heard':
        testcsv = 'metadata/vggss_heard_test.csv'
    elif args.testset == 'vggss_unheard':
        testcsv = 'metadata/vggss_unheard_test.csv'

    else:
        raise NotImplementedError
    bbox_format = {'flickr': 'flickr',
                   'flickr_plus_silent': 'flickr',
                   'vggss': 'vggss',
                   'vggss_plus_silent': 'vggss',
                   'vggss_heard': 'vggss',
                   'vggss_unheard': 'vggss'}[args.testset]

    #  Retrieve list of audio and video files
    testset = set([item[0] for item in csv.reader(open(testcsv))])

    # Intersect with available files
    audio_files = {fn.split('.wav')[0] for fn in os.listdir(audio_path)}
    image_files = {fn.split('.jpg')[0] for fn in os.listdir(image_path)}
    avail_files = audio_files.intersection(image_files)
    testset = testset.intersection(avail_files)

    testset = sorted(list(testset))
    image_files = [dt+'.jpg' for dt in testset]
    audio_files = [dt+'.wav' for dt in testset]

    # Bounding boxes
    all_bboxes = load_all_bboxes(args.test_gt_path, format=bbox_format)
    all_bboxes = [all_bboxes[fn.split('.jpg')[0]] for fn in image_files]

    if 'num_test_samples' in vars(args) and args.num_test_samples is not None and args.num_test_samples > 0 and len(image_files) > args.num_test_samples:
        idx = random.sample(range(len(image_files)), k=args.num_test_samples)
        image_files = [image_files[i] for i in idx]
        audio_files = [audio_files[i] for i in idx]
        all_bboxes = {fn.split('.')[0]: all_bboxes[fn.split('.')[0]] for fn in image_files}
    
    # load non-sounding files
    if args.testset in ['flickr_plus_silent', 'vggss_plus_silent']:
        name_testset = args.testset.split('_')[0]
        for item in csv.reader(open(f'metadata/{name_testset}_test_plus_silent.csv')):
            if item[2] == 'non-sounding':
                image_files.append(f'{item[0]}.jpg')
                audio_files.append(f'{item[1]}.wav')
                all_bboxes.append([])

        idx = list(range(len(image_files)))
        random.shuffle(idx)
        image_files = [image_files[i] for i in idx]
        audio_files = [audio_files[i] for i in idx]
        all_bboxes = [all_bboxes[i] for i in idx]

    # Transforms
    image_transform = transforms.Compose([
        transforms.Resize((224, 224), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    audio_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.0], std=[12.0])])

    return AudioVisualDataset(
        mode='test',
        image_files=image_files,
        audio_files=audio_files,
        image_path=image_path,
        audio_path=audio_path,
        audio_dur=5.,
        image_transform=image_transform,
        audio_transform=audio_transform,
        all_bboxes=all_bboxes,
        bbox_format=bbox_format
    )


def inverse_normalize(tensor):
    inverse_mean = [-0.485/0.229, -0.456/0.224, -0.406/0.225]
    inverse_std = [1.0/0.229, 1.0/0.224, 1.0/0.225]
    tensor = transforms.Normalize(inverse_mean, inverse_std)(tensor)
    return tensor

def convert_normalize(tensor, new_mean, new_std):
    raw_mean = IMAGENET_DEFAULT_MEAN
    raw_std = IMAGENET_DEFAULT_STD
    # inverse_normalize with raw mean & raw std
    inverse_mean = [-mean/std for mean, std in zip(raw_mean, raw_std)]
    inverse_std = [1.0/std for std in raw_std]
    tensor = transforms.Normalize(inverse_mean, inverse_std)(tensor)
    # normalize with new mean & new std
    tensor = transforms.Normalize(new_mean, new_std)(tensor)
    return tensor