# Paper published

| Dataset | P@10 | P@20 | R@10 | R@20 | N@10 | N@20 | H@10 | H@20 |
| - | -: | -: | -: | -: | -: | -: | -: | -: |
| TikTok   |            - |       0.0057 |         - |    0.1139 |       - |  0.0492 |           - |           - |
| Clothing |            - |       0.0045 |         - |    0.0881 |       - |  0.0394 |           - |           - |
| Sports   |            - |       0.0056 |         - |    0.1064 |       - |  0.0501 |           - |           - |



# Keep ori  (NVIDIA RTX PRO 6000)

| Dataset |   P@10 |   P@20 |   R@10 |   R@20 |   N@10 |   N@20 | H@10 | H@20 | MAP@10 | MAP@20 |
| ------- | -----: | -----: | -----: | -----: | -----: | -----: | ---: | ---: | -----: | -----: |
| TikTok  | 0.0073 | 0.0055 | 0.0727 | 0.1097 | 0.0388 | 0.0482 |    - |    - | 0.0285 | 0.0311 |
| Clothing| 0.0060 | 0.0045 | 0.0587 | 0.0878 | 0.0313 | 0.0386 |    - |    - | 0.0229 | 0.0249 |
| Sports  | 0.0077 | 0.0056 | 0.0728 | 0.1067 | 0.0400 | 0.0486 |    - |    - | 0.0295 | 0.0319 |
# Aligned MMRec--Evaluation Protocol (NVIDIA RTX PRO 6000)

| Dataset |   P@10 |   P@20 |   R@10 |   R@20 |   N@10 |   N@20 | H@10 | H@20 | MAP@10 | MAP@20 |
| ------- | -----: | -----: | -----: | -----: | -----: | -----: | ---: | ---: | -----: | -----: |
| TikTok  | 0.0073 | 0.0054 | 0.0730 | 0.1086 | 0.0376 | 0.0466 |    - |    - | 0.0269 | 0.0294 |
| Clothing  | 0.0059 | 0.0045 | 0.0585 | 0.0887 | 0.0315 | 0.0391 |    - |    - | 0.0232 | 0.0252 |
| Sports      | 0.0076 | 0.0057 | 0.0725 | 0.1074 | 0.0400 | 0.0489 |    - |    - | 0.0297 | 0.0321 |



# MMHCL-MM

This directory rewrites MMHCL into the MMRec/LGMRec-style framework:

- `main.py`: MMRec-style entrance.
- `models/mmhcl.py`: MMHCL model, including UI graph propagation, U-U graph propagation, I-I multimodal hypergraph construction, BPR loss, and user/item contrastive loss.
- `common/`: abstract recommender and trainer copied from the LGMRec-style framework, with feature loading extended for MMHCL audio features.
- `utils/`: dataset, dataloader, config, logger, metrics, and evaluator utilities.
- `configs/`: global, model, and dataset YAML configs.
- `data/*/*.inter`: converted interaction files reused from the original MMHCL conversion logs.
- `data/*/*_feat.npy`: copied modal feature files, so this project can run independently.

Dataset configs use project-local paths:

```yaml
data_path: './data/'
feature_data_path: './data/'
```

So interactions and modal features are both loaded from `MMHCL-MM/data/{dataset}/`.

Expected files:

- `data/clothing/clothing.inter`, `image_feat.npy`, `text_feat.npy`
- `data/sports/sports.inter`, `image_feat.npy`, `text_feat.npy`
- `data/tiktok/tiktok.inter`, `image_feat.npy`, `text_feat.npy`, `audio_feat.npy`

Run from this directory:

```bash
python main.py -m MMHCL -d tiktok -g 0
python main.py -m MMHCL -d clothing -g 0
python main.py -m MMHCL -d sports -g 0
```


