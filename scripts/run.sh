CUDA_VISIBLE_DEVICES=0 python3 main.py --gin_config_file=configs/ml-1m/hstu-sampled-softmax-n128-large-final.gin --master_port=12340
CUDA_VISIBLE_DEVICES=1 python3 main.py --gin_config_file=configs/ml-1m/hstu-sampled-softmax-n128-large-final.gin --master_port=12341
CUDA_VISIBLE_DEVICES=2 python3 main.py --gin_config_file=configs/ml-1m/hstu-sampled-softmax-n128-large-final-mix.gin --master_port=12342
CUDA_VISIBLE_DEVICES=3 python3 main.py --gin_config_file=configs/ml-1m/hstu-sampled-softmax-n128-large-final-text.gin --master_port=12345
CUDA_VISIBLE_DEVICES=3 python3 main.py --gin_config_file=configs/ml-1m/hstu-sampled-softmax-n128-large-final-domain-gating.gin --master_port=12345

CUDA_VISIBLE_DEVICES=0 python3 main.py --gin_config_file=configs/ml-20m/hstu-sampled-softmax-n128-large-final.gin --master_port=12340
CUDA_VISIBLE_DEVICES=1 python3 main.py --gin_config_file=configs/ml-20m/hstu-sampled-softmax-n128-large-final.gin --master_port=12341
CUDA_VISIBLE_DEVICES=2 python3 main.py --gin_config_file=configs/ml-20m/hstu-sampled-softmax-n128-large-final-text.gin --master_port=12342
CUDA_VISIBLE_DEVICES=3 python3 main.py --gin_config_file=configs/ml-20m/hstu-sampled-softmax-n128-large-final-mix.gin --master_port=12343