pip install -r requirements.txt

#Set your own PACS Dataset download path
./scripts/download_pacs.sh $PACS_DESIRED_ROOT

python -m torch.distributed.launch --nproc_per_node=1 main.py --dg --target cartoon --config_file configs/dg/dg.json --data_root $PACS_DESIRED_ROOT --name cartoon_exps_dg


#target cartoon
python -m torch.distributed.launch --nproc_per_node=1 main.py --dg --target cartoon --config_file configs/dg/dg.json --data_root ./dataset --name cartoon_exps_dg --runs 5

#target sketch
python -m torch.distributed.launch --nproc_per_node=1 main.py --dg --target sketch --config_file configs/dg/dg.json --data_root ./dataset --name sketch_exps_dg --runs 5


#target art
python -m torch.distributed.launch --nproc_per_node=1 main.py --dg --target art_painting --config_file configs/dg/dg.json --data_root ./dataset --name art_painting_exps_dg --runs 5


#target photo
python -m torch.distributed.launch --nproc_per_node=1 main.py --dg --target photo --config_file configs/dg/dg.json --data_root ./dataset --name photo_exps_dg --runs 5
