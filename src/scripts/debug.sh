shots=(
#  1
  5
)

classifiers=(
  "simpleshot"
  "laplacianshot"
  "tim"
  "alpha_tim"
  "ilpc"
  "om"
  "siamese"
)

embeddings=(
  "none"
  "l2"
  "cl2"
  "zn"
  "rr"
  "tcpr"
  "nohub"
  "nohubs"
)

wandb off

# Loop shots, classifiers and embeddings and run evaluation on the debug dataset.
for shot in "${shots[@]}"; do
  for classifier in "${classifiers[@]}"; do
    for embedding in "${embeddings[@]}"; do
      echo "----------------------------------------------------------------------------------------------------"
      echo "Running: shot=$shot, classifier=$classifier, embedding=$embedding"
      echo "----------------------------------------------------------------------------------------------------"
      python evaluate.py -c config/templates/debug.yml --n_shots $shot --classifier $classifier --embedding $embedding
      if [ $? != 0 ]; then
        # Abort if we got an exception from python
        exit $?
      fi
    done
  done
done
