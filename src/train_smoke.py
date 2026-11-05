"""抽烟行为二分类训练（MobileNetV3-small 迁移学习）并导出 ONNX。

数据来自 src/collect_smoke.py 采集的 dataset/smoke（labels.csv + rois/*.png）。
样本是「手-嘴联合 ROI」的 128×128 灰度裁剪，裁剪规则见 hand_crop.py，
训练与板端推理必须共用同一套裁剪。

用法（Mac 本机，CPU 足够）：
  ../.venv-train/bin/python src/train_smoke.py --data dataset/smoke --epochs 15

产物：
  models/smoke.onnx      板端推理用（输入 1x3x128x128，输出 2 类 logits）
  models/smoke.txt       训练信息（验证集指标）

划分方式：每个阶段按时间排序，前 70% 训练、后 30% 验证。相邻帧高度相似，
必须按时间切分而不是随机切分，否则验证分数会虚高。
"""

import argparse
import csv
import os
from collections import defaultdict

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import models, transforms

from hand_crop import NORM_MEAN, NORM_STD, SMOKE_SIZE

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGENET = dict(mean=NORM_MEAN, std=NORM_STD)


class SmokeSet(Dataset):
    def __init__(self, rows, train: bool) -> None:
        self.rows = rows
        if train:
            # 只有左右镜像、轻微平移缩放和亮度扰动。不做大角度旋转：
            # ROI 里嘴与手的上下关系是判别信息，转多了会造出假样本。
            self.tf = transforms.Compose([
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomAffine(degrees=6, translate=(0.05, 0.05), scale=(0.92, 1.08)),
                transforms.ColorJitter(brightness=0.15, contrast=0.15),
                transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 0.6))], p=0.1),
                transforms.ToTensor(),
                transforms.Normalize(**IMAGENET),
            ])
        else:
            self.tf = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize(**IMAGENET)])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = cv2.imread(r["abs"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((SMOKE_SIZE, SMOKE_SIZE), np.uint8)
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return self.tf(Image.fromarray(rgb)), int(r["label"])


def build_model() -> nn.Module:
    try:
        net = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    except Exception:  # 老版本 torchvision
        net = models.mobilenet_v3_small(pretrained=True)
    in_f = net.classifier[3].in_features
    net.classifier[3] = nn.Linear(in_f, 2)
    return net


def load_rows(data_dir: str):
    with open(os.path.join(data_dir, "labels.csv"), newline="") as f:
        rows = [dict(r) for r in csv.DictReader(f)]
    by_key = defaultdict(list)
    for r in rows:
        r["abs"] = os.path.join(data_dir, r["path"])
        r["ts"] = float(r["ts"])
        by_key[r["key"]].append(r)
    train, val = [], []
    for key in sorted(by_key):
        items = sorted(by_key[key], key=lambda r: r["ts"])
        # 按采集会话（间隔 >10 分钟）切段：训练集要覆盖所有机位/角度，
        # 验证集只用最新一轮采集的后 40%。否则模型只在旧角度上学，换个
        # 座位就全错——之前 hand_mouth 验证只有 0.21 就是这个原因。
        sess, cur = [], [items[0]]
        for r in items[1:]:
            if r["ts"] - cur[-1]["ts"] > 600:
                sess.append(cur)
                cur = []
            cur.append(r)
        sess.append(cur)
        if len(sess) >= 2:
            newest = sess[-1]
            cut = max(1, int(len(newest) * 0.6))
            train += [r for s in sess[:-1] for r in s] + newest[:cut]
            val += newest[cut:]
        else:
            cut = int(len(items) * 0.7)
            train += items[:cut]
            val += items[cut:]
    return train, val


@torch.no_grad()
def evaluate(net, loader, device):
    net.eval()
    preds, gts = [], []
    for x, y in loader:
        preds.append(net(x.to(device)).argmax(1).cpu())
        gts.append(y)
    p = torch.cat(preds).numpy()
    g = torch.cat(gts).numpy()
    acc = float((p == g).mean())
    tp = int(((p == 1) & (g == 1)).sum())
    fp = int(((p == 1) & (g == 0)).sum())
    fn = int(((p == 0) & (g == 1)).sum())
    rec = tp / max(1, tp + fn)
    prec = tp / max(1, tp + fp)
    f1 = 2 * prec * rec / max(1e-6, prec + rec)
    return dict(acc=acc, recall=rec, precision=prec, f1=f1, tp=tp, fp=fp, fn=fn), p, g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "dataset", "smoke"))
    ap.add_argument("--out", default=os.path.join(ROOT, "models", "smoke.onnx"))
    ap.add_argument("--epochs", type=int, default=15)
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
    dl_tr = DataLoader(SmokeSet(train_rows, True), batch_size=args.batch,
                       shuffle=True, num_workers=0)
    dl_va = DataLoader(SmokeSet(val_rows, False), batch_size=args.batch, num_workers=0)

    net = build_model().to(device)
    # 正负样本不均衡（约 1:2.4），按类别频率加权，避免模型偏向多数类
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
              f"smoke recall {m['recall']:.3f} prec {m['precision']:.3f} f1 {m['f1']:.3f}{flag}")

    net.load_state_dict(best[1])
    m, p, g = evaluate(net, dl_va, device)
    print(f"\nbest val: acc {m['acc']:.3f} recall {m['recall']:.3f} "
          f"precision {m['precision']:.3f} f1 {m['f1']:.3f}")
    print(f"confusion: TP {m['tp']}  FP {m['fp']}  FN {m['fn']}")

    # 按阶段看误报/漏报分布
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
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"\nexported {args.out} ({size_mb:.1f} MB)")

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
