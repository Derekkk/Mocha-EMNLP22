## Mocha-EMNLP22

This repository contains data and code for MOCHA in our EMNLP 2022 paper: [MOCHA: A Multi-Task Training Approach for Coherent Text Generation
from Cognitive Perspective](https://arxiv.org/pdf/2210.14650v1.pdf)

### Requirements

The original code is tested under the following environment:

```
pytorch==1.7.1
transformers==4.8.2
```

### Code Structure
- `finetune_generation_pipeline.py`: the code for training and decoding
- `run_pipeline.sh`: runing script

To run the code, you need to specify the model path and data path in `run_pipeline.sh`, and then run code with the command:
```
bash run_pipeline.sh
```
