# 384 / 十把不同密钥 / α=0.001

新实验；保留原同钥十次实验。预先固定 seed=101,103,…,119，使用奇数避免 key_seed|1 合并。
构建阶段只导出公开数据，必须确认十个完整公钥 SHA256 不同才启动。
每组独立按公钥搜索探针，不复用旧 key 的探针；搜索失败按失败保留，不换 key 重试到成功。
宽参数公开探针检查沿用当前算法，不宣称完成 3^16 向量的穷举不变性证明。

每组串行准备探针、六核采集 16×1664 个系数、公钥格纠错、独立 KEM/AEAD 验收，最后单独生成测试真值审计。
SPRT α=.001，Nmin=32，每16次检查，Nmax=1000；到上限仍给猜测，保留证据分数，强制猜测优先纠错。
不使用 vote3，也不复用 .01 的早停数据。α不是强制猜测或全密钥错误率保证。

后处理沿用原稀疏嵌入格统一六档预算；使用隔离的 BLASter GH 半径修正版。
本次比较同时改变了密钥、α和求解器修复版本，不能把任何变化单独归因于α。
记录 probes、capture、recovery、audit 及每30秒心跳；每组探针、采集、后处理时间分别统计。
主成功标准是完整公钥方程与随机 KEM/AEAD 验收，不读真值调参。

从仓库根目录运行：
```
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/yu/workspace/BLASter/venv/bin/python experiments/sprt-384-distinct10-a0001-20261101/build.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/yu/workspace/BLASter/venv/bin/python experiments/sprt-384-distinct10-a0001-20261101/run.py
```
拒绝覆盖旧构建或结果目录。失败不代表密钥被恢复，完整运行也不等于完整恢复。
