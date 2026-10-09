# CMSG 复现前检查 / Reproduction status

本文记录公开仓库中可核实的复现条件，回应 [复现求助 #3](https://github.com/Clouds1997/CMSG/issues/3)。它不是已经验证成功的训练或评估教程。

## 模型源码可以从历史版本找回

当前 `main` 的提交 `077001b87bed6893144f2f22ec61c4b1bf78b66b` 删除了 `models/` 目录，但 `kitti/train_seclassifier.py` 仍导入其中的 `Classifergraph` 等类。因此仅安装依赖不能让当前版本完整运行。

前一个提交 `d046f3fccd4e4384d778e4fcca54deb3216ee54c` 保留了模型源码。可以在独立目录查看该版本，避免改变已有工作目录：

```sh
git clone https://github.com/Clouds1997/CMSG.git CMSG-reproduction
cd CMSG-reproduction
git switch --detach d046f3fccd4e4384d778e4fcca54deb3216ee54c
```

该版本包含 `models/multimodal_classifier.py` 及模型的其他 Python/CUDA 源文件。恢复源码不等于恢复训练权重，也不能据此保证复现论文结果。

## 训练权重仍需原作者提供

检查公开 Git 历史未发现 `.pth`、`.pt`、`.ckpt` 或 `.safetensors` 文件。代码中提及的 `checkpoints/best.pth` 是本地路径，不是下载链接。不要将随机初始化模型或其他项目的权重当作 CMSG 的已训练模型。

为继续处理 #3，需要原作者提供或确认：

- 对应 CMSG 实验的 checkpoint 下载地址及对应代码版本。
- checkpoint 对应的模型类和配置；例如训练入口当前使用 `Classifergraph`，不能假定任意模型权重都兼容。
- KITTI 数据划分、语义数据预处理方法，以及可运行的训练/评估命令和依赖版本。

在这些材料齐全前，本文不提供推测的安装命令或可保证结果的训练命令。

## 本次验证范围

已从上述历史提交导出 `models/`，并用 Python `ast.parse` 检查其中 33 个 Python 文件，均可解析。未运行训练、CUDA 扩展或论文指标评估；语法检查不代表运行兼容性验证。

English: model source is recoverable from the pinned historical commit, but no trained checkpoint was found in the public Git history. Reproducing the reported results still requires the author's checkpoint, matching configuration, data preparation, and environment details.
