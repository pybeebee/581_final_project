python train.py --task=emotion --data=mfcc --model=svm
python train.py --task=intensity --data=mfcc --model=svm
python train.py --task=emotion --data=gfcc --model=svm
python train.py --task=intensity --data=gfcc --model=svm

python train.py --task=emotion --data=mfcc --model=logreg
python train.py --task=intensity --data=mfcc --model=logreg
python train.py --task=emotion --data=gfcc --model=logreg
python train.py --task=intensity --data=gfcc --model=logreg

python train.py --task=emotion --data=mfcc --model=perceptron
python train.py --task=intensity --data=mfcc --model=perceptron
python train.py --task=emotion --data=gfcc --model=perceptron
python train.py --task=intensity --data=gfcc --model=perceptron

python train.py --task=emotion --data=mfcc --model=mlp
python train.py --task=intensity --data=mfcc --model=mlp
python train.py --task=emotion --data=gfcc --model=mlp
python train.py --task=intensity --data=gfcc --model=mlp