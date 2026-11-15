"""Render completed full capture wall time, cost and independent acceptance."""
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'scripts'))
from sequential_oracle import evidence


def main():
    out=HERE/'full'
    run=json.loads((out/'results.json').read_text())
    if run['status']!='complete' or not run['input_hashes_verified']:raise ValueError('full run not complete')
    for path,digest in run['inputs_sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise ValueError(f'input changed: {path}')
    rows=list(csv.DictReader(line for line in (out/'capture.log').read_text().splitlines()
                             if not line.startswith('#')))
    ns=np.array([[int(r['n_a']),int(r['n_b'])] for r in rows])
    buckets=np.array([[int(r['tb_base']),int(r['tb_shift'])] for r in rows])
    sampled=ns>0
    stopped=(buckets!=255)&sampled
    undecided=(buckets==255)&sampled
    counts=dict(axes=int(ns.size),stopped_axes=int(stopped.sum()),unresolved_axes=int(undecided.sum()),
                constant_axes=int((~sampled).sum()),
                accepted_coefficients=sum(int(r['guess'])!=99 for r in rows),
                abstained_coefficients=run['capture_abstentions'])
    unresolved_coefficients=(buckets==255).any(axis=1)
    reference_rejected=np.array([r['ref_ok']=='0' for r in rows])
    breakdown=dict(unresolved_only=int((unresolved_coefficients&~reference_rejected).sum()),
                   reference_rejected_only=int((reference_rejected&~unresolved_coefficients).sum()),
                   both_unresolved_and_reference_rejected=int((reference_rejected&unresolved_coefficients).sum()),
                   accepted=counts['accepted_coefficients'])
    if sum(breakdown.values())!=13024:raise ValueError('abstention reasons do not reconcile')
    (out/'abstention-breakdown.json').write_text(json.dumps(breakdown,indent=2)+'\n')
    # Count saved raw samples independently of the collector's counters.
    observed=np.zeros((11,1184,2),dtype=np.int64)
    indexed={(int(r['coord']),int(r['position'])):r for r in rows}
    def verify_sequence(key,samples):
        row=indexed[key]
        for axis,name in enumerate(('a','b')):
            values=np.asarray(samples[axis])
            public=[int(row[f'{name}_{label}']) for label in ('neg','zero','pos')]
            low,high=min(public),max(public)
            actual=int(row['tb_base' if axis==0 else 'tb_shift'])
            if low==high:
                if len(values) or actual!=low:raise ValueError('constant axis misclassified')
                continue
            boundary=(.5*(low+high)-int(row['ref_bucket']))*float(row['gap_ns'])
            logs=evidence(values,boundary)
            looks=np.arange(len(values))+1
            eligible=(looks>=run['min_samples'])&((looks%run['batch_samples']==0)|(looks==run['max_samples']))
            cross=np.flatnonzero(eligible&(logs.max(axis=1)>=np.log(2/run['alpha_per_axis'])))
            if len(cross):
                stop=int(cross[0])+1
                predicted=high if logs[stop-1,1]>logs[stop-1,0] else low
                if stop!=len(values) or actual!=predicted:raise ValueError('live stopping differs from replay')
            elif len(values)!=run['max_samples'] or actual!=255:
                raise ValueError('unresolved axis did not exhaust budget')
            if not np.isclose(logs[-1].max(),float(row[f'loge_{name}']),rtol=1e-10,atol=1e-9):
                raise ValueError('logged mixture evidence differs from samples')
    for path in sorted(out.glob('raw-*.csv')):
        current=None;samples=[[],[]]
        with path.open() as stream:
            for line in stream:
                col,pos,axis,n,value=line.rstrip().split(',')
                key=(int(col),int(pos),int(axis))
                if int(n)!=observed[key]+1:raise ValueError('raw sample sequence gap')
                observed[key]+=1
                if key[:2]!=current:
                    if current is not None:verify_sequence(current,samples)
                    current=key[:2];samples=[[],[]]
                samples[key[2]].append(float(value))
        if current is not None:verify_sequence(current,samples)
    for row in rows:
        key=(int(row['coord']),int(row['position']))
        if observed[key].tolist()!=[int(row['n_a']),int(row['n_b'])]:raise ValueError('raw sample counts differ')
    counts['raw_counts_verified']=True
    counts['live_stopping_and_evidence_verified']=True
    (out/'sampling-summary.json').write_text(json.dumps(counts,indent=2)+'\n')
    table='\n'.join(f"| {c['column']} | {c['initial_wrong_non_erased']} | {c['erasures']} | "
                    f"{'通过' if c['public_valid'] else '未通过'} |" for c in run['columns'])
    costs=run['timing_costs']
    accepted=counts['accepted_coefficients'];wrong=run['initial_wrong_non_erased']
    report=f'''# 完整一组 256：动态 N 的在线 SPRT

完成全部 **13,024 个系数**的单次真实采集，不做 vote3。
物理采集墙钟 **{run['physical_capture_seconds']:.1f} 秒（{run['physical_capture_seconds']/60:.2f} 分钟）**；
含公开探针重新验证的采集进程耗时 **{run['capture_process_seconds']:.1f} 秒**。
后续公钥纠错 **{run['correction_seconds']:.1f} 秒**；全部流程 **{run['total_seconds']:.1f} 秒
（{run['total_seconds']/60:.2f} 分钟）**。时间是实际运行值，不是前缀回放或按小样本外推。

公钥独立验收 **{run['public_accepted_columns']}/11 列**。
本轮{'完整恢复，另通过 KEM/AES-GCM/负例验证' if run['public_accepted_columns']==11 else '未完整恢复'}。
{'未接受任何完整列，不能以空集合的零错数宣称恢复正确。' if run['public_accepted_columns']==0 else f"接受列事后真值错数为 {run['accepted_truth_mismatches']}。"}

## SPRT 和实际 N

使用符号 Bernoulli 混合 SPRT：α=0.001/轴，双侧阈值 2/α；最低 32 对样本，
每批 16 对，上限 1000。两轴分别停止，到上限仍未定则弃权。
实测每轴平均 **{run['mean_samples_per_axis']:.1f} 对**；
N 的最小/中位/P90/P95/P99/最大为 **{run['sample_quantiles']}**。
{counts['stopped_axes']} 个轴达到证据阈值，{counts['unresolved_axes']} 个轴未决，
另 {counts['constant_axes']} 个轴的公开桶恒定，不需要测量。
达上限计数 {run['capped_axes']} 包括恰在最后一批接受的情况，不能当成未决数。

六个 worker 固定 CPU 1/2/4/5/7/6，RDTSCP+LFENCE，C 内随机相邻配对。
每 worker 初始预热与三轮校准，每处理 128 个系数重新校准，每系数另检查 32 对
局部参考。局部参考拒绝 {run['reference_rejected_coefficients']} 个系数。
这对参考稳定性的检查不保证参考模板与每个目标的计时分布一致。

## 成本及准确率

真实计时解封装共 **{run['timing_calls']:,} 次**：

| 项目 | 调用数 |
|---|---:|
| 预热 | {costs['warmup']:,} |
| 校准及重试 | {costs['calibration']:,} |
| 目标和配对参考 | {costs['target_and_paired_reference']:,} |
| 局部参考 | {costs['local_reference']:,} |

采集输出 {accepted} 个非弃权猜测，其中事后审计发现 **{wrong} 个错误**；
另 **{run['capture_abstentions']} 个弃权**。非弃权错误率为
**{100*wrong/accepted if accepted else float('nan'):.3f}%**。
α 不是该错误率的后验保证，也不是全密钥错误率上界；边界偏移或相关噪声可能导致
高证据误判。样本总数已与逐 worker 原始 CSV 独立逐条核对，且每个轴的首次停止
批次、接受方向和累计证据均经独立 Python 实现回放核对。

弃权按互斥原因拆分：**{breakdown['unresolved_only']}** 个仅因为 SPRT 未决，
**{breakdown['reference_rejected_only']}** 个已达到 SPRT 阈值但局部参考被拒，
**{breakdown['both_unresolved_and_reference_rejected']}** 个同时满足两种原因。
因此 4,123 个弃权不能全部算作 SPRT 未决。零错误接受是本组有限样本的观察，
未证明未来数据零误判或满足 α 对应的桶错误率。

公钥纠错共纠正 **{run['corrected_wrong_non_erased']} 个错误猜测**、补全
**{run['completed_erasures']} 个弃权**。只接受满足全部公钥方程的三值列。

| 列（从 0 起） | 采集错误猜测 | 弃权 | 公钥验收 |
|---:|---:|---:|---|
{table}

## 比较范围及复现

同密钥历史固定 N=100 单次采集约 1255.5 秒，初始 472 个错误猜测、77 个弃权，
较小纠错预算恢复 6/11 列。当前加入动态停止、周期校准、局部参考检查，纠错另含
360/640 支持集，因此不能把前后差异全部归因于 SPRT，也不能宣称严格同预算胜出。
未重新执行固定 N 对照；复用了公开探针缓存，耗时不含重新发现探针的成本。

采集器不计算真实桶或解包 S；求解只读新采集的可观察字段、公钥。
全部列完成求解并独立公钥验收后才读历史同公钥日志的真值进行审计。
`capture.log` 的 truth/精确桶列为占位符，准确率以 `results.json` 的独立审计为准。

协议见 `../PROTOCOL.md`，源码 `../build.py`、`../sprt.h`、`../run.py`，
构建哈希 `../build.json`，日志 `capture.err`、`capture.log`，原始差分 `raw-*.csv`。
`public-acceptance.json` 在真值审计前保存；完整成功时另有 `verification.json`。
`sampling-summary.json` 给出按实际停止状态统计的计数。

```sh
python experiments/sprt-256-20261029/build.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/yu/workspace/BLASter/venv/bin/python experiments/sprt-256-20261029/run.py
```

运行要求 `full/` 不存在，避免覆盖已有结果。当前运行已完成，复现请复制实验脚本到新的实验目录。
'''
    (out/'REPORT.md').write_text(report)
    print(json.dumps({**counts,**{k:run[k] for k in ('physical_capture_seconds','total_seconds','timing_calls',
          'public_accepted_columns','initial_wrong_non_erased','capture_abstentions')}},indent=2))


if __name__=='__main__':main()
