# MobileNetV3
A Keras implementation of MobileNetV3 and Lite R-ASPP Semantic Segmentation (Under Development).

According to the paper: [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244?context=cs)

## Requirement
- Python 3.6
- Tensorflow-gpu 1.10.0  
- Keras 2.2.4

## Train the model

 The ```config/config.json``` file provide a config for training.

### Train the classification

**The dataset folder structure is as follows:**

	| - data/
		| - train/
	  		| - class 0/
				| - image.jpg
					....
			| - class 1/
			  ....
			| - class n/
		| - validation/
	  		| - class 0/
			| - class 1/
			  ....
			| - class n/

**Run command below to train the model:**

```
python train_cls.py
```

## Acknowledgement

Thank [@fzyzcjy](https://github.com/fzyzcjy) for your help in this project.

## Reference

	@article{MobileNetv3,  
	  title={Searching for MobileNetV3},  
	  author={Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang Vijay Vasudevan, Quoc V. Le, Hartwig Adam},
	  journal={arXiv preprint arXiv:1905.02244},
	  year={2019}
	}


## Copyright
See [LICENSE](LICENSE) for details.
