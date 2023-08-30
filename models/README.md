

OverpassNL - Model Training
=====================================================

Python Version: python==3.10

Training:
```python train_t5.py --exp_name default --data_dir ../dataset --model_name google/byt5-base  --gradient_accumulation_steps 4```

Inference:
```python inference_t5.py --exp_name default --model_name byt5-base --data_dir ../dataset --num_beams 4```
