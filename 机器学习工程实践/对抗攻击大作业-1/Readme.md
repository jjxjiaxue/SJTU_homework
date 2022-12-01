## Linf attack

- 攻击resnet
```bash
CUDA_VISIBLE_DEVICES=0 python pgd_linf.py -lp <label_path> -ip <image_path>
```

- 攻击vit
```bash
CUDA_VISIBLE_DEVICES=0 python pgd_linf.py -a v -lp <label_path> -ip <image_path>
```

## L2 attack

- 攻击resnet
```bash
CUDA_VISIBLE_DEVICES=0 python pgd_l2.py -lp <label_path> -ip <image_path>
```

- 攻击vit
```bash
CUDA_VISIBLE_DEVICES=0 python pgd_l2.py -a v -lp <label_path> -ip <image_path>
```

## Patch attack

- 攻击resnet
```bash
CUDA_VISIBLE_DEVICES=0 python pgd_patch.py -lp <label_path> -ip <image_path>
```

- 攻击vit
```bash
CUDA_VISIBLE_DEVICES=0 python pgd_patch.py -a v -lp <label_path> -ip <image_path>
```

## Transfer Attack

- 使用vit对抗样本攻击resnet
```bash
CUDA_VISIBLE_DEVICES=0 python pgd_transfer.py -a v -lp <label_path> -ip <image_path>
```

- 使用resnet对抗样本攻击vit
```bash
CUDA_VISIBLE_DEVICES=0 python pgd_transfer.py -lp <label_path> -ip <image_path>
```

## Black Box L2 Attack

- 攻击resnet
```bash
CUDA_VISIBLE_DEVICES=0 python black_box.py -lp <label_path> -ip <image_path>
```

- 攻击vit
```bash
CUDA_VISIBLE_DEVICES=0 python black_box.py -a v -lp <label_path> -ip <image_path>
```

