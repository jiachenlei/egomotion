# egomotion

We, as team "**TheSSVL**" or "**EgoMotion-COMPASS**", took **2nd** place in both **Object State Change Classification** and **PNR temporal localization** tasks in Ego4d Challenge 2022  


### Moreover, **our work on Egocentric video understanding** will be made publicly available by Nov 2022.  



TODO

- [ ] Post Techincal report on Arxiv  
- [x] Release codes which we used in Ego4d Challenge 2022  
- [ ] Release codes of our latest work on egocentric video understading  

### Environment requirements
in addition to "wandb", we use same environment as [VideoMAE](https://github.com/MCG-NJU/VideoMAE) and [ego4d oscc i3d-resnet50 baseline](https://github.com/EGO4D/hands-and-objects/tree/main/state-change-localization-classification/i3d-resnet50). Please refer to the repos for more information

### Usage

1. Finetuning pretrained weights on Ego4d oscc and temporal localization at the same time:

- Modify required paramters including dataset path in config/finetune_vitb_ego4d.yml or config/finetune_vitl_ego4d.yml, e.g.
```
    finetune: "" # path to the pretrained weight
```
ps: you can download pretrained videoMAE weights from videoMAE repository:[vitl](https://drive.google.com/file/d/1qLOXWb_MGEvaI7tvuAe94CV7S2HXRwT3/view?usp=sharing), [vitb](https://drive.google.com/file/d/1JfrhN144Hdg7we213H1WxwR3lGYOlmIn/view?usp=sharing)

- Modify required paramters in ./scripts/finetune_ego4d.sh

- Finally, in ./scripts, run

```python
    # finetune on single node 
    bash finetune_ego4d.sh 0 0.0.0.0

    # finetune on two nodes:
    # run on first node
    bash finetune_ego4d.sh 0 0.0.0.0
    # run on second node
    bash finetune_ego4d.sh 1 ip_address_of_first_machine
```

2. Test on Ego4d oscc and temporal localization:

- Similar to 1, modify required paramters including dataset path in config/test_ego4d.yml

- Modify required paramters in ./scripts/test_ego4d.sh

- Finally, in ./scripts, run

```python
    # test on single node 
    bash test_ego4d.sh 0 0.0.0.0

    # test on two nodes:
    # run on first node
    bash test_ego4d.sh 0 0.0.0.0
    # run on second node
    bash test_ego4d.sh 1 ip_address_of_first_machine
```

For emphasis, two json files (one for oscc one for temporal localization) in the format specified by ego4d challenge will be generated and stored in the directory:
```
$output_dir/$name
```
where $output_dir is specified in config/test_ego4d.yml
and $name is specified in test_ego4d.sh

### Reference
[1] VideoMAE by Zhan, etc : [VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training](https://arxiv.org/abs/2203.12602)  
[2] VideoMAE by Kaiming, etc : [Masked Autoencoders As Spatiotemporal Learners](https://arxiv.org/abs/2205.09113)  
[3] Vanilla MAE: [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377)  
[4] Ego4D: [Ego4D: Around the World in 3,000 Hours of Egocentric Video](https://arxiv.org/abs/2110.07058)  


### Contact
If you have any questions about our projects or implementation, please open an issue or contact via email:  
Jiachen Lei: jiachenlei@zju.edu.cn 

### Acknowledgements
We built our codes based on [VideoMAE](https://github.com/MCG-NJU/VideoMAE), [MAE-pytorch](https://github.com/pengzhiliang/MAE-pytorch). Thanks to all the contributors of these great repositories.
