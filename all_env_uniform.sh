DIR="experiment_runs/uniform"

for TYPE in normal naive
do
    for MODEL in bc dt rvs
    do
        for ENV in MO-Hopper-v2 MO-Swimmer-v2 MO-Ant-v2 MO-HalfCheetah-v2 MO-Walker2d-v2 MO-Hopper-v3
        do
            if [ "$MODEL" = "rvs" ]; then
                STEP=100000
                ITER=2
                CSP=1
                CAP=0
                CRP=0
            elif [ "$MODEL" = "dt" ]; then
                ITER=2
                CSP=1
                CAP=1
                CRP=1
                if [ "$ENV" = "MO-Hopper-v2" ] || [ "$ENV" = "MO-Hopper-v3" ]; then
                    STEP=200000
                elif [ "$ENV" = "MO-Swimmer-v2" ]; then
                    STEP=130000
                elif [ "$ENV" = "MO-Ant-v2" ]; then
                    STEP=10000
                elif [ "$ENV" = "MO-HalfCheetah-v2" ]; then
                    STEP=40000
                elif [ "$ENV" = "MO-Walker2d-v2" ]; then
                    STEP=180000
                fi
            else # else, model must be bc
                STEP=100000
                ITER=2
                CSP=1
                CAP=0
                CRP=0
            fi
            # if naive, change respective concat
            if [ "$TYPE" = "naive" ]; then
                CSP=0
                CAP=0
                CRP=0
            fi
            echo "***" $TYPE $MODEL $ENV $CSP $CRP $CAP $STEP "***"
            CUDA_VISIBLE_DEVICES=1 python experiment.py --dir $DIR --env $ENV --data_mode _formal --concat_state_pref $CSP --concat_rtg_pref $CRP --concat_act_pref $CAP --mo_rtg True --seed 1 --dataset expert_uniform --model_type $MODEL --num_steps_per_iter $STEP --max_iters $ITER &
            CUDA_VISIBLE_DEVICES=1 python experiment.py --dir $DIR --env $ENV --data_mode _formal --concat_state_pref $CSP --concat_rtg_pref $CRP --concat_act_pref $CAP --mo_rtg True --seed 2 --dataset expert_uniform --model_type $MODEL --num_steps_per_iter $STEP --max_iters $ITER &
            CUDA_VISIBLE_DEVICES=1 python experiment.py --dir $DIR --env $ENV --data_mode _formal --concat_state_pref $CSP --concat_rtg_pref $CRP --concat_act_pref $CAP --mo_rtg True --seed 3 --dataset expert_uniform --model_type $MODEL --num_steps_per_iter $STEP --max_iters $ITER &
            CUDA_VISIBLE_DEVICES=2 python experiment.py --dir $DIR --env $ENV --data_mode _formal --concat_state_pref $CSP --concat_rtg_pref $CRP --concat_act_pref $CAP --mo_rtg True --seed 4 --dataset amateur_uniform --model_type $MODEL --num_steps_per_iter $STEP --max_iters $ITER &
            CUDA_VISIBLE_DEVICES=2 python experiment.py --dir $DIR --env $ENV --data_mode _formal --concat_state_pref $CSP --concat_rtg_pref $CRP --concat_act_pref $CAP --mo_rtg True --seed 5 --dataset amateur_uniform --model_type $MODEL --num_steps_per_iter $STEP --max_iters $ITER &
            CUDA_VISIBLE_DEVICES=2 python experiment.py --dir $DIR --env $ENV --data_mode _formal --concat_state_pref $CSP --concat_rtg_pref $CRP --concat_act_pref $CAP --mo_rtg True --seed 6 --dataset amateur_uniform --model_type $MODEL --num_steps_per_iter $STEP --max_iters $ITER &
            wait
        done
    done
done
