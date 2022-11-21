task="cmv"
is_mtl="true"

# seeds=(1001 1002 1003 1004 1005)
# few_shot_rates=(0.01 0.05 0.1 0.2 0.3)

seeds=(1234)
few_shot_rates=(-1)


t5_path="./model_card/t5"

# distributed train
for few_shot_rate in "${few_shot_rates[@]}"
do
    for seed in "${seeds[@]}"
    do
        #output_dir="save_model/cmv/seed"${seed}"_fewshot"${few_shot_rate}"_mtl_"${is_mtl}
        output_dir="save_model/cmv_t5"
        echo $output_dir

        CUDA_VISIBLE_DEVICES=2,3 \
        python3 -m torch.distributed.launch \
            --nproc_per_node=2 \
            --nnodes=1 \
            --master_port=9901 \
            finetune_generation_pipeline.py \
                --seed ${seed} \
                --few_shot_rate ${few_shot_rate} \
                --epochs 18 \
                --batch_size 8 \
                --save_step -1 \
                --output_dir ${output_dir} \
                --tokenizer_path ${t5_path} \
                --model_path ${t5_path} \
                --task ${task} \
                --is_mtl ${is_mtl} >save_model/log-cmv-mtl 2>&1 &
    done
done
