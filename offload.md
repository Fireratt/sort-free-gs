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