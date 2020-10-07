# -*- coding: utf-8 -*-
import argparse
import os
import torch
import json
import torch.backends.cudnn as cudnn
from torch import distributed as distributed
from data.mat_dataset import MatDataset
from data.domain_dataset import PACSDataset, DomainDataset
from methods import CuMix
from utils import test
import numpy as np
import pickle
from tqdm import tqdm


ZSL_DATASETS = ['CUB', 'FLO', 'SUN', 'AWA1']
PACS_DOMAINS = ['photo', 'art_painting', 'cartoon', 'sketch']
DNET_DOMAINS = ['real', 'quickdraw', 'sketch', 'painting', 'infograph', 'clipart']

parser = argparse.ArgumentParser(description='Zero-shot Learning meets Domain Generalization -- ZSL experiments')
parser.add_argument('--target', default='cub', help='Which experiment to run (e.g. [cub, awa1, flo, sun, all])')
parser.add_argument('--zsl', action='store_true', help='ZSL setting?')
parser.add_argument('--dg', action='store_true', help='DG setting?')
parser.add_argument('--data_root', default='./data/xlsa17/data', type=str, help='Data root directory')
parser.add_argument('--name', default='test', type=str, help='Name of the experiment (used to store '
                                                           'the logger and the checkpoints)')
parser.add_argument('--runs', default=10, type=int, help='Number of runs per experiment')
parser.add_argument('--log_dir', default='./logs', type=str, help='Log directory')
parser.add_argument('--ckpt_dir', default='./checkpoints', type=str, help='Checkpoint directory')
parser.add_argument('--config_file', default=None, help='Config file for the method.')
parser.add_argument("--local_rank", type=int, default=0)

args = parser.parse_args()

# # 실험 세팅

distributed.init_process_group(backend='nccl', init_method='env://')
device_id, device = args.local_rank, torch.device(args.local_rank)
rank, world_size = distributed.get_rank(), distributed.get_world_size()
torch.cuda.set_device(device_id)
world_info = {'world_size':world_size, 'rank':rank}


# Check association dataset--setting are correct + init remaining stuffs from configs
assert args.dg or args.zsl, "Please specify if you want to benchmark ZSL and/or DG performances"

config_file = args.config_file
with open(config_file) as json_file:
    configs = json.load(json_file)
print(args.config_file)
multi_domain = False
input_dim = 2048
configs['freeze_bn'] = False

# Semantic W is used to rescale the principal semantic loss.
# Needed to have the same baseline results as https://github.com/HAHA-DL/Episodic-DG/tree/master/PACS for DG only exps
semantic_w = 1.0


if args.dg:
    target = args.target
    multi_domain = True
    if args.zsl:
        assert args.target in DNET_DOMAINS, \
            args.target + " is not a valid target domain in  DomainNet. Please specify a valid DomainNet target in " + DNET_DOMAINS.__str__()
        DOMAINS = DNET_DOMAINS
        dataset = DomainDataset
    #DG Setting일경우 
    else:
        assert args.target in PACS_DOMAINS, \
            args.target + " is not a valid target domain in PACS. Please specify a valid PACS target in " + PACS_DOMAINS.__str__()
        #PACS_DOMAINS 종류 리스트 DOMAINS 변수에 저장
        DOMAINS = PACS_DOMAINS
        #PACSDataset 클래스 dataset 변수에 저장
        dataset = PACSDataset
        input_dim = 512
        semantic_w = 3.0
        configs['freeze_bn']=True

    #sources라는 soruce domains 저장하는 리스트 생성
    sources = DOMAINS + []
    ns = len(sources)
    #sources에서 target인 도메인 빼기
    sources.remove(target)
    assert len(sources) < ns, 'Something is wrong, no source domains reduction after remove with target.'
else:
    target = args.target.upper()
    assert target in ZSL_DATASETS, \
        args.target + " is not a valid ZSL dataset. Please specify a valid dataset " + ZSL_DATASETS.__str__()
    sources = target
    dataset = MatDataset
    configs['mixup_img_w'] = 0.0
    configs['iters_per_epoch'] = 'max'

configs['input_dim'] = input_dim
configs['semantic_w'] = semantic_w
configs['multi_domain'] = multi_domain

# Init loggers and checkpoints path
log_dir = args.log_dir
checkpoint_dir = args.ckpt_dir
exp_name = args.name
cudnn.benchmark = True


exp_name=args.name+'.pkl'


try:
    os.makedirs(log_dir)
except OSError:
    pass

try:
    os.makedirs(checkpoint_dir)
except OSError:
    pass

log_file = os.path.join(log_dir, exp_name)
if os.path.exists(log_file):
    print("WARNING: Your experiment logger seems to exist. Change the name to avoid unwanted overwriting.")

checkpoint_file = os.path.join(checkpoint_dir, args.name + '-runN.pth')
if os.path.exists(checkpoint_file):
    print("WARNING: Your experiment checkpoint seems to exist. Change the name to avoid unwanted overwriting.")

logger = {'results':[], 'config': configs, 'target': target, 'checkpoints':[], 'sem_loss':[[] for _ in range(args.runs)],
          'mimg_loss':[[] for _ in range(args.runs)], 'mfeat_loss':[[] for _ in range(args.runs)]}
results = []
results_top = []
val_datasets = None

# # 학습 및 테스트

# +
# Start experiments loop
#학습 시작 

#runs만큼 여러번 돌리기
for r in range(args.runs):
    print('\nTarget: ' + target + '    run ' +str(r+1) +'/'+str(args.runs))

    ##데이터 불러오기 
    # Create datasets
    #train, val, test dataset 모두 같은 transformer 사용
    train_dataset = dataset(args.data_root, sources,train=True)
    test_dataset = dataset(args.data_root, target, train=False)
    if args.dg and not args.zsl:
        val_datasets = []
        #source domain들로 validation하기
        for s in sources:
            val_datasets.append(dataset(args.data_root, s, train=False, validation=True))

    #데이터의 attributes 담기?
    attributes = train_dataset.full_attributes
    #PACS 기준 (seen : 7)
    seen = train_dataset.seen
    #PACS 기준 (unseen : 7)
    unseen = train_dataset.unseen

    # Init method
    ##CuMIx 모델 설정
    method = CuMix(seen_classes=seen,unseen_classes=unseen,attributes=attributes,configs=configs,zsl_only = not args.dg,
                   dg_only = not args.zsl,device=device,world_size=world_size,rank=rank)

    #?
    #Epoch마다 테스트한 모델의 성능 
    temp_results = []
    #validation set에서 가장 높은 모델의 정확도
    top_sources = 0.
    top_idx=-1

    # Strat training loop
    #training 시작 
    #tqdm : Python용 진행표시바 표시하기 
    for e in tqdm(range(0, configs['epochs'])):
        #training 시작 
            #semantic_loss : classification loss
            #mimg_loss : image-level mixup classification loss
            #mfeat_loss : feature-level mixup classification loss
        semantic_loss, mimg_loss, mfeat_loss = method.fit(train_dataset)
        #모델 test
        accuracy = test(method, test_dataset, zsl=args.zsl)


        # In case of DG only, perform validation on source domains, as in https://arxiv.org/pdf/1710.03077.pdf
        if val_datasets is not None:
            acc_sources = 0.
            for v in val_datasets:
                acc_sources += test(method, v, device, zsl=False)
                #source domain에 대한 평균적인 정확도 계산 
                acc_sources /= len(sources)
                #현재까지의 정확도보다 더크면 결과 갱신 
                if acc_sources > top_sources:
                    top_sources = acc_sources
                    temp_results = accuracy
        #validation dataset이 없으면, 단순히 test dataset의 결과를 현재 성능으로 저장 
        else:
            temp_results = accuracy

        # Store losses
        #lossses log에 저장 
        logger['sem_loss'][r].append(semantic_loss)
        logger['mimg_loss'][r].append(mimg_loss)
        logger['mfeat_loss'][r].append(mfeat_loss)

    #runs 만큼 다 돌면 모델 가중치 파일 저장 및 
    # Store checkpoints and update logger
    checkpoint_dict = {}
    method.save(checkpoint_dict)
    current_checkpoint_name = checkpoint_file.replace('runN.pth','run'+str(r+1)+'.pth')
    torch.save(checkpoint_dict, current_checkpoint_name)

    logger['results'].append(temp_results)
    logger['checkpoints'].append(current_checkpoint_name)
    print(target,logger['results'][top_idx])
# -


#runs만큼 돌린 결과들에 대한 평균하고 분산 출력
print('\nResults for ' + target, np.mean(logger['results']),np.std(logger['results']))

with open(log_file, 'wb') as handle:
    #Python 객체 저장 
    pickle.dump(logger, handle, protocol=pickle.HIGHEST_PROTOCOL)
