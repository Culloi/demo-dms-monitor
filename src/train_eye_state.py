"""眼睛睁/闭二分类训练（MobileNetV3-small 迁移学习）并导出 ONNX。

数据来自 src/collect_eyes.py 采集的 dataset/eyes（labels.csv + crops/*.png）。

用法（Mac 本机，CPU 足够）：
  .venv-train/bin/python src/train_eye_state.py --data dataset/eyes --epochs 15

产物：
  models/eye_state.onnx      板端推理用（输入 1x3x96x96，输出 2 类 logits）
  models/eye_state.txt       训练信息（验证集指标、阈值建议）

划分方式：按「阶段 × 采集会话」切分（间隔 >30s 视为新会话），每个会话内部时间
前 80% 训练、后 20% 验证。相邻帧高度相似，不能随机切分；而按阶段整体时间切会让
后补的会话整体落进验证集，模型永远学不到新姿势。
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

from eye_crop import EYE_SIZE, NORM_MEAN, NORM_STD

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGENET = dict(mean=NORM_MEAN, std=NORM_STD)


class EyeSet(Dataset):
    def __init__(self, rows, train: bool) -> None:
        self.rows = rows
        if train:
            self.tf = transforms.Compose([
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomAffine(degrees=8, translate=(0.06, 0.06), scale=(0.9, 1.1)),
                transforms.ColorJitter(brightness=0.3, contrast=0.3),
                transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 1.0))], p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(**IMAGENET),
                transforms.RandomErasing(p=0.15, scale=(0.02, 0.08)),
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
            img = np.zeros((EYE_SIZE, EYE_SIZE), np.uint8)
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


def split_sessions(items, gap: float = 30.0):
    """把同一阶段内按时间排序的样本按采集会话切开（相邻会话间隔远大于帧间隔）。"""
    out, cur, last = [], [], None
    for r in items:
        if last is not None and r["ts"] - last > gap:
            out.append(cur)
            cur = []
        cur.append(r)
        last = r["ts"]
    if cur:
        out.append(cur)
    return out


def load_rows(data_dir: str, gap: float = 30.0):
    """按「阶段 × 采集会话」切分：每个会话内部按时间前 80% 训练 / 后 20% 验证。

    不能按阶段整体时间切：新补的会话时间一定更晚，会整体落进验证集，模型只会
    在旧姿势上学，换坐姿/换眼睑放松程度就整体失效（抽烟模型踩过同一个坑）。
    """
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
        for si, sess in enumerate(split_sessions(items, gap)):
            cut = int(len(sess) * 0.8)
            for r in sess:
                r["sess"] = f"{key}#{si}"
            train += sess[:cut]
            val += sess[cut:]
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
    ap.add_argument("--data", default=os.path.join(ROOT, "dataset", "eyes"))
    ap.add_argument("--out", default=os.path.join(ROOT, "models", "eye_state.onnx"))
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    train_rows, val_rows = load_rows(args.data)
    print(f"train {len(train_rows)}  val {len(val_rows)}  device {device}")
    dl_tr = DataLoader(EyeSet(train_rows, True), batch_size=args.batch,
                       shuffle=True, num_workers=0)
    dl_va = DataLoader(EyeSet(val_rows, False), batch_size=args.batch, num_workers=0)

    net = build_model().to(device)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
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
              f"closed recall {m['recall']:.3f} prec {m['precision']:.3f} f1 {m['f1']:.3f}{flag}")

    net.load_state_dict(best[1])
    m, p, g = evaluate(net, dl_va, device)
    print(f"\nbest val: acc {m['acc']:.3f} recall {m['recall']:.3f} "
          f"precision {m['precision']:.3f} f1 {m['f1']:.3f}")
    print(f"confusion: TP {m['tp']}  FP {m['fp']}  FN {m['fn']}")

    # 按「阶段 × 会话」看：哪一段数据（旧姿势/新姿势）更容易错
    for sess in sorted({r["sess"] for r in val_rows}):
        idx = [i for i, r in enumerate(val_rows) if r["sess"] == sess]
        if not idx:
            continue
        # 睁眼误判为闭眼的占比，是板端最关心的指标
        open_idx = [i for i in idx if val_rows[i]["label"] == 0]
        fp_rate = (float((p[open_idx] == 1).mean()) if open_idx else float("nan"))
        print(f"  {sess:<20} acc {float((p[idx] == g[idx]).mean()):.3f}  "
              f"open->closed {fp_rate:.3f}  n={len(idx)}")

    net.eval().cpu()
    dummy = torch.randn(1, 3, EYE_SIZE, EYE_SIZE)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.onnx.export(net, dummy, args.out, input_names=["input"],
                      output_names=["logits"], opset_version=13,
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}})
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"\nexported {args.out} ({size_mb:.1f} MB)")

    # 用 onnxruntime 复核导出结果与 PyTorch 一致
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
                f"input=1x3x{EYE_SIZE}x{EYE_SIZE} imagenet_norm\n"
                f"val acc={m['acc']:.4f} recall={m['recall']:.4f} "
                f"precision={m['precision']:.4f} f1={m['f1']:.4f}\n"
                f"confusion TP={m['tp']} FP={m['fp']} FN={m['fn']}\n")


if __name__ == "__main__":
    main()
