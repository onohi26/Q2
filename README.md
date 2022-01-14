# Q2

[Geometric matching CNN](https://arxiv.org/abs/1703.05593)と[VoxelMorph](https://arxiv.org/abs/1809.05231v3)により位置合わせを行いました。

# Installation

必要なライブラリをインストールしてください。

 ```
 pip install -r requirements.txt
 ```

# Preparation

`images_source`と`images_target`は`./datasets/`ディレクトリ以下に配置してください。

# Step 1

Geometric matching CNNによりアフィン変換を行います。

モデルの学習は以下のように行ってください。

```
python train_affine.py
```

テストは以下のように行ってください。

```
python demo_affine.py
```

テスト時に勾配降下法により、モデルの最適化を行うため、テストには時間がかかります。

テストの結果画像は`./datasets/result_affine`に、キーポイント座標は`./datasets/affine.json`に出力されます。

結果画像とキーポイント座標はVoxelMorphのテストの際に必要です。

# Step 2

VoxelMorphにより非線形の変換を行います。

モデルの学習は以下のように行ってください。

```
python train_vxm.py
```

テストは以下のように行ってください。

このときGeometric matching CNNにより得られた結果画像とキーポイント座標を使用します。

```
python demo_vxm.py
```

テスト時に勾配降下法により、モデルの最適化を行うため、テストには時間がかかります。

テストの結果画像は`./datasets/result_vxm`に、キーポイント座標は`vxm.json`に出力されます。