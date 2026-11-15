# Scloud+ paper artifact

本仓库固定保存论文摘要采用的三组 Scloud+ 实验：对应历史源码、完整原始计时、采集日志、恢复结果、公钥及新鲜密文验证记录。它是独立仓库，不依赖旧仓库的 Git 历史。

> Our attacks recover full Scloud+ secret keys at all three vulnerable parameter sets: 10/10 independent keys at L256 and L384, averaging 105.7 and 162.1 minutes, respectively, and one L512 key in 519.5 minutes.

| 参数 | 密钥 | 在线采集时间 | 实验版本 | 对应源码 |
|---|---:|---:|---|---|
| L256 | 10/10，seed 101, 103, …, 119 | 平均 105.7 分钟 | 原 binary sign-mixture SPRT，α=0.001 | [source/l256](source/l256)，历史提交 `429a5bd` |
| L384 | 10/10，seed 101, 103, …, 119 | 平均 162.1 分钟 | 原 template SPRT，α=0.001 | [source/l384](source/l384)，历史提交 `1a44a19` |
| L512 | 1/1，seed=1 | 519.5 分钟 | 2026-10-30 binary SPRT，α=0.01 | [source/l512](source/l512)，历史提交 `429a5bd` |

在线时间不包含离线探针搜索、重建、格纠错或本地验证。L512 后续 α=0.001 模板实验与 L384 binary replay 不属于本仓库的结果。公开输入和探针预测参与恢复；真实私钥标签只用于事后审计。

## 查阅结果

- [实验与版本索引](experiment-index.json)：各档参数、源码版本、归档路径与摘要数字。
- [results](results)：21 个原始实验的摘要、恢复记录和原始验证结果，内容与归档逐字节一致。
- [results-verification.json](reports/results-verification.json)：由原始记录重新汇总的时间、调用数和成功数量。
- [fresh-verification](reports/fresh-verification)：本次整理时，使用归档中的恢复矩阵和公钥重新进行的公钥方程、随机新鲜 KEM 密文和 AEAD 验证。此步骤没有重新采集计时，不是新的攻击性能实验。
- [archives](archives)：全部原始数据，压缩后约 1.8 GiB，直接由 Git 保存，每个分卷不超过 10 MiB；无需 Git LFS 或外部下载。
- [provenance](provenance)：源文件 SHA256、原始文件映射、历史提交及论文版本绑定。

## 校验和解压

需要 Linux、Python 3.11+ 和 `zstd`。仅核对归档完整性无需额外 Python 包。

```sh
# 校验全部分卷及每一个解压文件的 SHA256，不写出原始数据。
python3 tools/archive.py

# 解压到新的目录，完整数据约 5.8 GiB；原始相对目录结构保留。
python3 tools/archive.py --extract .work/data

# 可只读取一档：
python3 tools/archive.py --level 512 --extract .work/only512
```

L256 与 L384 分卷和清单原样复制。L512 从历史共享归档中仅选出论文使用的 seed=1 数据、夹具、构建材料和共同源码重新打包；保留原始成员路径和文件字节，每个文件均核对恢复清单的 SHA256。旧 L384 α=0.01 实验的原始数据没有混入。

## 编译和独立验证

需要 Linux x86-64、GCC，以及支持 AVX2、AES-NI、BMI2、POPCNT 的 CPU。以下命令编译冻结的原始采集器和公钥验证器，不开始耗时的攻击采集。

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python3 tools/build.py

# 在完成上面的解压和编译后，对全部 21 个已恢复密钥进行新鲜验证。
python3 tools/verify_results.py --fresh-kem
```

`tools/build.py` 使用各自历史源码和原来的编译选项，不通过当前统一入口选择算法；因此 L512 不会被误替换成模板版本。生成的二进制、编译命令和日志在 `.work/build/`，不会写回冻结源码。重新编译的二进制可能因编译器不同而与历史二进制不同。

`tools/verify_results.py` 只使用保存的恢复矩阵和公开输入进行新鲜验证，不读取真实秘密标签。核验内容包含全部公钥方程、随机新鲜 KEM 共享密钥一致性、AEAD 解密、篡改拒绝及错误恢复矩阵拒绝。

## 源码范围和重新运行

`source/l256`、`source/l384`、`source/l512` 保留各自历史目录结构、采集器、判决代码、离线恢复、格纠错、公钥验证及 Scloud+ 实现依赖；源文件未经改写。各目录下的 `experiments/.../PROTOCOL.md` 和原始 runner 记录对应协议。L256 复用了既有公开探针夹具，所以该夹具作为依赖一并保留；它不构成额外计入的实验结果。

原始 runner 含有历史机器路径、旧实验目录引用和核绑定（1/2/4/5/6/7）。完整重跑前需在独立工作目录中恢复所需夹具并适配路径和 CPU 核；不要直接在冻结目录运行会创建/覆盖文件的历史构建脚本。本次已验证原始采集器可编译，未重新执行数小时的在线攻击采集。

[BLASter](vendor/blaster-fixed) 是格纠错依赖的保留源码，附原始许可证。它来自保留运行仓库 `bc2cb2d`；此来源单独记录，不将其称为历史二进制的重建证明。若需运行格约简，可安装 Eigen3、Cython、cysignals、setuptools 后构建此扩展，并将其 `src` 加入 `PYTHONPATH`。历史 Python 工具的其余导入依赖应按实际使用安装。

## 统计口径

“纠错前”指离线评分和最佳猜测已经完成、格纠错尚未开始的估计。L256 的 10 把密钥和论文 L512 的一次运行在这个阶段为零错误，但仍包含未达到可信停止条件的猜测；不能将其解释成在线 SPRT 确定了每一个系数。L384 有一把密钥留下五个错误，随后由格纠错修复。完整流程均通过最终公钥及新鲜密文验证。

历史 JSON 中的绝对路径仅作为原始证据保留；阅读和校验应使用本仓库的相对路径及工具。后续实验应另建目录，不自动替换本仓库的论文基准。
