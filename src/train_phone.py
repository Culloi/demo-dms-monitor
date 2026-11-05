"""打电话行为二分类训练（MobileNetV3-small 迁移学习）并导出 ONNX。

数据来自 src/collect_phone.py 采集的 dataset/phone（labels.csv + rois/*.png），
样本是「手-耳联合 ROI」的 128×128 灰度裁剪，裁剪规则见 phone_crop.py，
训练与板端推理必须共用同一套裁剪。

用法（Mac 本机，CPU 足够）：
  ../.venv-train/bin/python src/train_phone.py --data dataset/phone --epochs 25

产物：
  models/phone.onnx      板端推理用（输入 1x3x128x128，输出 2 类 logits）
  models/phone.txt       训练信息（验证集指标）

划分方式与抽烟一致：按采集会话切分，训练集覆盖所有机位/角度，
验证集只用最新一轮采集的后 40%。
"""

import argparse
import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from hand_crop import SMOKE_SIZE
from train_smoke import SmokeSet as RoiSet, build_model, evaluate, load_rows

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "dataset", "phone"))
    ap.add_argument("--out", default=os.path.join(ROOT, "models", "phone.onnx"))
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    train_rows, val_rows = load_rows(args.data)
    n_pos = sum(1 for r in train_rows if int(r["label"]) == 1)
    print(f"train {len(train_rows)} (pos {n_pos})  val {len(val_rows)}  device {device}")
    dl_tr = DataLoader(RoiSet(train_rows, True), batch_size=args.batch,
                       shuffle=True, num_workers=0)
    dl_va = DataLoader(RoiSet(val_rows, False), batch_size=args.batch, num_workers=0)

    net = build_model().to(device)
    w_neg = len(train_rows) / max(1, 2 * (len(train_rows) - n_pos))
    w_pos = len(train_rows) / max(1, 2 * n_pos)
    crit = nn.CrossEntropyLoss(weight=torch.tensor([w_neg, w_pos], dtype=torch.float32).to(device),
                               label_smoothing=0.05)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best = (-1.0, None)
    for ep in range(1, args.epochs + 1):
        net.train()
        total, seen = 0.0, 0
        for x, y in dl_tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(net(x), y)
            loss.backward()
            opt.step()
            total += float(loss) * len(y)
            seen += len(y)
        sched.step()
        m, _, _ = evaluate(net, dl_va, device)
        score = m["f1"] * 0.7 + m["acc"] * 0.3
        flag = ""
        if score > best[0]:
            best = (score, {k: v.detach().clone() if torch.is_tensor(v) else v
                            for k, v in net.state_dict().items()})
            flag = " *"
        print(f"ep{ep:02d} loss {total / max(1, seen):.4f} | val acc {m['acc']:.3f} "
              f"phone recall {m['recall']:.3f} prec {m['precision']:.3f} f1 {m['f1']:.3f}{flag}")

    net.load_state_dict(best[1])
    m, p, g = evaluate(net, dl_va, device)
    print(f"\nbest val: acc {m['acc']:.3f} recall {m['recall']:.3f} "
          f"precision {m['precision']:.3f} f1 {m['f1']:.3f}")
    print(f"confusion: TP {m['tp']}  FP {m['fp']}  FN {m['fn']}")

    for key in sorted({r["key"] for r in val_rows}):
        idx = [i for i, r in enumerate(val_rows) if r["key"] == key]
        if not idx:
            continue
        print(f"  {key:<14} acc {float((p[idx] == g[idx]).mean()):.3f}  n={len(idx)}")

    net.eval().cpu()
    dummy = torch.randn(1, 3, SMOKE_SIZE, SMOKE_SIZE)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.onnx.export(net, dummy, args.out, input_names=["input"],
                      output_names=["logits"], opset_version=13,
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}})
    print(f"\nexported {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)")

    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
        x, _ = next(iter(dl_va))
        ref = net(x).detach().numpy()
        got = sess.run(None, {"input": x.numpy()})[0]
        print(f"onnxruntime check: max diff {float(np.abs(ref - got).max()):.2e}")
    except Exception as e:
        print(f"onnxruntime check skipped: {e}")

    with open(os.path.splitext(args.out)[0] + ".txt", "w") as f:
        f.write(f"data={args.data}\ntrain={len(train_rows)} val={len(val_rows)}\n"
                f"input=1x3x{SMOKE_SIZE}x{SMOKE_SIZE} imagenet_norm\n"
                f"val acc={m['acc']:.4f} recall={m['recall']:.4f} "
                f"precision={m['precision']:.4f} f1={m['f1']:.4f}\n"
                f"confusion TP={m['tp']} FP={m['fp']} FN={m['fn']}\n")


if __name__ == "__main__":
    main()
