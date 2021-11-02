
declare -a trainsets=('objfacescenes' 'objects' 'faces' 'scenes')

for dset in ${trainsets[@]};
do
    COMMAND="python scripts/train_encoder.py --batchsize 256 --arch resnet50 --trainset ${dset}"
    echo $COMMAND
    sbatch --export="COMMAND=${COMMAND}" --job-name encoder --time 14-00:00:00 -p gpu --cpus-per-task=6 --gres=gpu:1 --mem=20G --output=log/%j.log scripts/run_slurm.sbatch
done