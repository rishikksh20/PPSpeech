# PPSpeech

### Training
```buildoutcfg
python train.py -o checkpoints -l logs --name "first" --config configs\default.yaml
```

### Inference
```buildoutcfg
python inference.py  -c "checkpoints\first\checkpoint_first_32000.pyt" -r "LJ002-0321.npy" -p "the invention of" -cu "movable metal letters in the middle of the fifteenth century  may justly be considered  as the invention" -po "of the art of printing " --config "configs\default.yaml"
```
