To train a model, run python model_train.py --model-out model_output/xgb_18x275_minQ2=1_29Jan2026.json --input-files rootfiles/train/18x275/minQ2\=1/*.root 
Edit input files and output name of course to match formate

To evaluate a model, run python model_evaluate.py python model_evaluate.py --model-in ....json --input-files ....root --output-prefix pictures/my_model_eval

