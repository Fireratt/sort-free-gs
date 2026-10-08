我建议你的实现顺序
Phase 1：Frozen 16M

先验证：

16M checkpoint
+
Sort-Free renderer
+
sigma/depth correction

GPU 只加载 inference parameters。

目标：

能 render
能计算 loss
Phase 2：Naive CPU Offload

实现：

CPU:
  all Gaussian
  optimizer state

GPU:
  full Gaussian
  gradient

每 iteration：

H2D
forward
backward
D2H
CPU optimizer

关闭：

densification
pruning

目标不是性能，而是：

16M 可以稳定训练
Phase 3：只优化真正需要的东西

如果 Phase 2 能跑，再考虑：

CPU→GPU transfer

是否成为瓶颈。

如果：

H2D = 300ms
training = 100ms

那才值得做：

prefetch
double buffering
pinned memory
overlap H2D / render / optimizer

优化4. 如果只是“微调 Sort-Free 参数”，可以直接 Freeze 16M Gaussian

这个我认为你们现在非常值得先试。

如果你们真正要优化的是 Sort-Free 的：

sigma
weight_background
...

而不是重新训练：

xyz
scale
rotation
opacity
SH

那么完全没必要让 16M Gaussian 参与 autograd。

例如：

with torch.no_grad():
    gaussian_features = ...

或者更彻底地：

for p in gaussian_model.parameters():
    p.requires_grad_(False)

sortfree_sigma.requires_grad_(True)
sortfree_background.requires_grad_(True)

这样会发生非常大的变化：

现在：

16M Gaussian
      ↓
forward
      ↓
autograd graph
      ↓
backward buffers
      ↓
gradient

变成：

16M Gaussian
      ↓
forward only
      ↓
small Sort-Free parameters
      ↓
backward
这对你当前实验尤其重要

因为你不是在做一个新的 16M Gaussian reconstruction。

你已经有：

Grendel → 16M Gaussian Building

现在只是：

在这个模型上 fine-tune Sort-Free parameters

如果 Gaussian 本身不需要更新，就应该把它们当成冻结的 scene data，而不是 trainable parameters。

这可能比任何 CPU offload 都有效。

该模式通过 `--freeze_gaussians` 启用；运行 `finetune_recursive.sh` 时可设置 `ENABLE_FREEZE_GAUSSIANS=1`。此模式冻结原始高斯的位置、颜色、不透明度、尺度和旋转，继续优化各向异性透明度 SH（`f_do`、`f_ro`）、`sigma` 和 `weight_background`。Rasterizer backward 只计算这些参数需要的梯度，跳过其他逐高斯梯度。

优化5.

目前除了各向异性透明度的sh参数之外的所有参数都不需要发生变化，因此我们可以认为在Tile-Level光栅化之前的所有stage，在所有iteration内都是不变的。因此需要常驻显存的只有部分高斯参数，而光栅化之前的所有Stage的计算结果都可以提前计算出来放在CPU内。比较简单的做法可能是为每个个相机各生成一个cpu侧的bitmap，对应目前我们所有高斯的可见性，再按需将他们传入GPU中做计算。bitmap在iteration中应该是静止不动的。以Building数据集为例，应该会有1920个bitmap，每个bitmap大小是2MB。

最终的整个流程变为：

读取全量模型； 进行Preprocess，完成视锥剔除 ； 为1920个相机均做一次，CPU做记录。 只对bitmap中记录的可见高斯做计算。
