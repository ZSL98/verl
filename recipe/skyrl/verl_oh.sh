set -x

export VLLM_ASCEND_ENABLE_NZ=0

DATA_ROOT=${DATA_ROOT:-$PWD}
DATA_DIR=$DATA_ROOT/data
train_data="${DATA_DIR}/swebench_lite_runnable/train.parquet"
test_data="${DATA_DIR}/swebench_lite_runnable/validation.parquet"


MODEL=/root/workspace/models/Qwen3-4B-Instruct-2507
NNODES=1
SP_SIZE=4
TP_SIZE=4
# NNODES=1
# SP_SIZE=2
# TP_SIZE=2

project_name=swe_agent
experiment_name=qwen3-4b_swe_agent_npu
default_local_dir=$DATA_ROOT/checkpoint/$experiment_name

# Get the directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
YAML_PATH="${SCRIPT_DIR}/verl_oh.yaml"

python3 -m recipe.skyrl.verl_main_ppo \
    algorithm.adv_estimator=rloo \
    data.train_files=$train_data \
    data.val_files=$test_data \
    data.dataloader_num_workers=0 \
    data.train_batch_size=8 \
    data.max_prompt_length=8000 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=true \
    actor_rollout_ref.model.path=$MODEL \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$SP_SIZE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP_SIZE \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    algorithm.norm_adv_by_std_in_grpo=False \
    trainer.val_before_train=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=$NNODES \
    trainer.max_actor_ckpt_to_keep=10 \
    trainer.save_freq=1 \
    trainer.default_local_dir=$default_local_dir \
    trainer.test_freq=20 \
    trainer.total_epochs=15 \
    trainer.rollout_data_dir="${DATA_DIR}/rollout_data_dir" \
    trainer.validation_data_dir="${DATA_DIR}/validation_data_dir" \
    +skyrl_agent.task_yaml="$YAML_PATH" \
    +skyrl_agent.num_trajectories=2 \
    trainer.device=npu $@