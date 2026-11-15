"""Validate and report capped decisions with confidence-directed correction."""
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def main():
    out=HERE/'soft-decision'
    run=json.loads((out/'results.json').read_text())
    baseline=json.loads((HERE/'full/results.json').read_text())
    before=json.loads((out/'public-acceptance.json').read_text())
    if run['status']!='complete' or not run['input_hashes_verified']:
        raise ValueError('wait for recovery and audit')
    for path,digest in run['inputs_sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:
            raise ValueError(f'input changed: {path}')
    with np.load(out/'public.npz',allow_pickle=False) as p:
        A,B=p['A'].astype(np.int64),p['B'].astype(np.int64)
    with np.load(out/'columns/input-state.npz',allow_pickle=False) as p:
        guesses,orders,erasures=p['guesses'],p['orders'],p['erasures']
        trusted,margin=p['trusted'],p['confidence_margin']
    if erasures.any() or not np.isin(guesses,[-1,0,1]).all():
        raise ValueError('soft decisions must assign every coefficient')
    scores=[json.loads(line) for line in (out/'confidence.jsonl').read_text().splitlines()]
    if len(scores)!=13024:raise ValueError('incomplete confidence records')
    for record in scores:
        key=(record['column'],record['position'])
        if guesses[key]!=record['guess'] or margin[key]!=record['margin']:
            raise ValueError('confidence records disagree with solver inputs')
    ranks=[]
    for record,accepted in zip(run['columns'],before['columns'],strict=True):
        col=record['column']
        if not accepted['public_valid'] or not record['public_valid']:
            raise ValueError('expected eleven independently accepted columns')
        value=np.load(out/'columns'/f'column-{col}.npy',allow_pickle=False)
        if not np.isin(value,[-1,0,1]).all() or np.any(abs((B[:,col]-A@value+512)%1024-512)>1):
            raise ValueError('public equation rejection')
        expected=np.lexsort((np.arange(1184),margin[col],trusted[col].astype(int)))
        if not np.array_equal(orders[col],expected):raise ValueError('confidence order mismatch')
        inverse=np.argsort(orders[col])
        for pos in np.flatnonzero(value!=guesses[col]):
            ranks.append(dict(column=col,position=int(pos),rank_1based=int(inverse[pos])+1,
                              before=int(guesses[col,pos]),after=int(value[pos]),
                              margin=float(margin[col,pos])))
    verification=json.loads((out/'verification.json').read_text())
    for key in ('fresh_kem_shared_secret_match','authenticated_message_decrypted',
                'tampered_aead_rejected','corrupted_secret_rejected'):
        if not verification[key]:raise ValueError(f'verification failed: {key}')
    if verification['public_equations_passed']!=13024:raise ValueError('incomplete public checks')
    summary=dict(public_accepted_columns=11,corrected_coefficients=len(ranks),
                 correction_ranks=ranks,additional_measurements=0,
                 postprocessing_seconds=run['total_seconds'],correction_seconds=run['correction_seconds'],
                 original_physical_capture_seconds=baseline['physical_capture_seconds'],
                 original_capture_process_seconds=baseline['capture_process_seconds'],
                 capture_plus_new_postprocessing_seconds=baseline['capture_process_seconds']+run['total_seconds'],
                 timing_caveat='capture plus postprocessing is a sum of measured stages, not a fresh contiguous run',
                 accepted_truth_mismatches=run['accepted_truth_mismatches'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    rows='\n'.join(f"| {c['column']} | {c['initial_wrong_non_erased']} | 通过 | {c['accepted_truth_mismatches']} |"
                   for c in run['columns'])
    rank_rows='\n'.join(f"| {r['column']} | {r['position']} | {r['before']} → {r['after']} | {r['rank_1based']} |"
                        for r in ranks)
    report=f'''# SPRT 到上限也输出猜测：按置信度优先纠错

**完整恢复 11/11 列**。全部 13,024 条公钥方程、随机 KEM 共享秘密一致、AES-GCM
解密、篡改密文拒绝和损坏秘密拒绝检查通过；验收后真值审计零错误。

复用上一轮全部原始测量，**没有新增问询，也没有 vote3**。保留原来 8,901 个
已接受猜测，对原来 4,123 个弃权位置也输出最佳三值猜测，并保留置信度及参考标志。
这 4,123 个位置中，事后审计仅 **{run['forced_guess_errors']} 个猜错**；原来已接受的
8,901 个仍有 **{run['trusted_guess_errors']} 个错误**。纠错前全矩阵仅 3 个错误，
随后全部纠正。因此之前的 0/11 完整列主要来自把大量已有方向信息当作未知值丢掉。

## 判决与置信度

逐轴读取实际采样到停止时的全部原始目标/参考差分，沿用固定边界与原混合 SPRT
公式，计算低桶、高桶的两侧 log evidence。对 −1/0/+1 三个合法候选，把对应的
两轴分数相加，选总分最大者；原来已经接受的猜测保留。
置信度保存为最佳候选与次优候选的分数差，同时保留三个候选分数、两轴分数、
原来是否弃权及局部参考是否有效。这个分数用于排序，**不是正确率后验概率**。

纠错顺序：原弃权位置先进入可修改支持集，其中证据差越小越优先；随后是原已接受
的位置，仍按证据差排序。位置编号仅用于打破平局。高置信度也未永久冻结，支持集
可逐步扩大。所有位置都有猜测，`erasures` 全为 false。
参考检查不通过仍输出猜测并降低优先保留等级，不再直接丢弃系数。

不再增加测量意味着这次没有把已经提前停止但参考被拒的轴补测到 N=1000；
它们使用原实际样本数。原 SPRT 未决轴本来就已用满 N=1000。
这是对已有实测数据的完整后处理验证；采集器的旧日志保持不变，新结果另存。

## 恢复结果

沿用上轮相同的窗口、方程、权重和单次约简预算；三列并行，每列单核。
**所有列在第一档 60 位置窗口就通过公钥校验**，没有用历史恢复列初始化。

| 列（从 0 起） | 重新判决后的错误 | 公钥校验 | 恢复后审计错误 |
|---:|---:|---|---:|
{rows}

实际纠正位置及其事先生成的纠错优先级如下，编号越小越先修改：

| 列 | 位置（从 0 起） | 修正 | 置信度排序（从 1 起） |
|---:|---:|---|---:|
{rank_rows}

排序生成和全部求解先完成，独立公钥验收写入 `public-acceptance.json` 后才读取
同一公钥历史真值做审计。上表位置来自验收后的比较，未反馈到求解。

## 耗时与成本

- 新后处理总耗时 **{run['total_seconds']:.2f} 秒**，含读取原始差分、重新判决、纠错和验收。
- 其中重新判决及准备约 **{run['soft_decision_seconds']:.2f} 秒**；公钥纠错 **{run['correction_seconds']:.2f} 秒**。
- 新增物理问询 **0 次**；沿用原采集 **22,429,280 次**计时解封装。
- 原物理采集 **110 分 58 秒**，含公开探针重新验证的采集进程 **{baseline['capture_process_seconds']/60:.2f} 分钟**。
- 原采集进程加本次新后处理的实测阶段之和约 **{summary['capture_plus_new_postprocessing_seconds']/60:.2f} 分钟**。
  这是已有采集与新后处理的耗时相加，不是重新连续跑过一遍的端到端计时；
  不包含之前弃权方案失败的 354.7 秒纠错。

本结果只证明这一组 256 的新处理方式成功，未验证全部十组，也未证明下一次
独立采集仍能零错或同样快速纠正。原始日志与旧失败结果完整保留。

## 文件与复现

`confidence.jsonl` 保存全部 13,024 个猜测及证据；`columns/input-state.npz` 保存
同一份猜测和排序；`results.json` 保存求解、耗时与事后审计；`verification.json`
保存完整 KEM/AEAD 验收；`summary.json` 保存独立复核与实际修正位置。

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/yu/workspace/BLASter/venv/bin/python experiments/sprt-256-20261029/run_soft_decision.py
```

输出目录必须不存在；复现时应使用新的输出路径，避免覆盖本轮产物。
新增三项测试覆盖未决时仍输出合法候选、证据强弱、参考拒绝降级和不合法桶对选择。
'''
    (out/'REPORT.md').write_text(report)
    snapshot=out/'source-snapshot';snapshot.mkdir(exist_ok=True)
    paths=[HERE/'run_soft_decision.py',Path(__file__),ROOT/'scripts/sprt_soft_decision.py',
           ROOT/'scripts/sequential_oracle.py',ROOT/'tests/test_sprt_soft_decision.py']
    for path in paths:shutil.copyfile(path,snapshot/path.name)
    (out/'artifact-manifest.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                                                        for p in paths},indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
