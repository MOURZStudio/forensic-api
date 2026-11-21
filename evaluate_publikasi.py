"""
evaluate_publikasi.py — uji batch data uji publikasi (Tabel 2).

Memakai fungsi yang SAMA dengan API (app_publikasi.analyze_bytes),
sehingga skor per citra identik dengan yang tampil di web.

Struktur folder (nama folder diawali nomor kelompok Tabel 2):
    data_uji/
      1_asli_kamera/        -> asli
      2_asli_whatsapp/      -> asli
      3_manipulasi_edit/    -> manipulasi
      4_asli_kaggle/        -> asli
      5_manipulasi_kaggle/  -> manipulasi
      6_manipulasi_ai/      -> manipulasi

Pemakaian:
    python evaluate_publikasi.py data_uji hasil/
Keluaran: hasil/hasil_per_citra.csv, hasil/ringkasan.md
"""
import csv
import os
import sys

from app_publikasi import analyze_bytes, W_M, W_C, T, VERSION

POSITIVE_GROUPS = {3, 5, 6}  # kelas positif = manipulasi
GROUP_NAMES = {
    1: 'Citra asli (kamera ponsel)',
    2: 'Citra asli via WhatsApp',
    3: 'Manipulasi editing software & metadata',
    4: 'Citra asli (Kaggle)',
    5: 'Citra manipulasi (Kaggle)',
    6: 'Manipulasi AI (faceswap & generate)',
}
EXT = ('.jpg', '.jpeg', '.png', '.webp', '.heic', '.bmp', '.tif', '.tiff')
SCEN = [('metadata_tunggal', 'Metadata tunggal'),
        ('cnn_tunggal', 'CNN tunggal'),
        ('gabungan', 'Gabungan (weighted)')]


def metrics(y_true, y_pred):
    tp = sum(t and p for t, p in zip(y_true, y_pred))
    tn = sum((not t) and (not p) for t, p in zip(y_true, y_pred))
    fp = sum((not t) and p for t, p in zip(y_true, y_pred))
    fn = sum(t and (not p) for t, p in zip(y_true, y_pred))
    n = tp + tn + fp + fn
    acc = (tp + tn) / n if n else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return dict(TP=tp, TN=tn, FP=fp, FN=fn, n=n,
                akurasi=acc, presisi=prec, recall=rec, f1=f1)


def mcnemar(y_true, pred_a, pred_b):
    """Uji McNemar eksak (binomial dua sisi) antara dua skenario."""
    from math import comb
    b = sum((pa == t) and (pb != t) for t, pa, pb in zip(y_true, pred_a, pred_b))
    c = sum((pa != t) and (pb == t) for t, pa, pb in zip(y_true, pred_a, pred_b))
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    p = 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return b, c, min(1.0, p)


def main(src, out):
    os.makedirs(out, exist_ok=True)
    rows = []
    for folder in sorted(os.listdir(src)):
        path = os.path.join(src, folder)
        if not os.path.isdir(path) or not folder[:1].isdigit():
            continue
        group = int(folder.split('_')[0])
        for fn in sorted(os.listdir(path)):
            if not fn.lower().endswith(EXT):
                continue
            with open(os.path.join(path, fn), 'rb') as f:
                r = analyze_bytes(f.read())
            m, c, w = r['metadata'], r['cnn'], r['weighted']
            rows.append({
                'kelompok': group, 'file': fn,
                'label': 'Manipulasi' if group in POSITIVE_GROUPS else 'Asli',
                'exif_count': m['exif_count'], 'c': m['c'], 'a': m['a'], 'b': m['b'],
                'S_m': round(m['score'], 4), 'S_c': round(c['score'], 4),
                'S': round(w['score'], 4),
                'pred_metadata': m['decision'], 'pred_cnn': c['decision'],
                'pred_gabungan': w['decision'],
            })
            print(f"[{group}] {fn}: S_m={m['score']:.4f} S_c={c['score']:.4f} "
                  f"S={w['score']:.4f} -> {w['decision']}")

    with open(os.path.join(out, 'hasil_per_citra.csv'), 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    y = [r['label'] == 'Manipulasi' for r in rows]
    preds = {
        'metadata_tunggal': [r['pred_metadata'] == 'Manipulasi' for r in rows],
        'cnn_tunggal': [r['pred_cnn'] == 'Manipulasi' for r in rows],
        'gabungan': [r['pred_gabungan'] == 'Manipulasi' for r in rows],
    }

    L = [f'# Ringkasan uji — {VERSION}',
         f'Bobot W_m = {W_M}, W_c = {W_C}, ambang T = {T} (S > T = manipulasi). '
         f'Jumlah citra: {len(rows)}.', '',
         '## Kinerja per skenario', '',
         '| Skenario | TP | TN | FP | FN | Akurasi | Presisi | Recall | F1-score |',
         '|---|---|---|---|---|---|---|---|---|']
    for key, name in SCEN:
        k = metrics(y, preds[key])
        L.append(f"| {name} | {k['TP']} | {k['TN']} | {k['FP']} | {k['FN']} | "
                 f"{k['akurasi']*100:.2f}% | {k['presisi']*100:.2f}% | "
                 f"{k['recall']*100:.2f}% | {k['f1']*100:.2f}% |")

    L += ['', '## Akurasi per kelompok citra', '',
          '| No | Kelompok | n | Metadata | CNN | Gabungan |', '|---|---|---|---|---|---|']
    for g in sorted({r['kelompok'] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r['kelompok'] == g]
        cells = []
        for key, _ in SCEN:
            ok = sum(preds[key][i] == y[i] for i in idx)
            cells.append(f'{ok}/{len(idx)} ({ok/len(idx)*100:.1f}%)')
        L.append(f"| {g} | {GROUP_NAMES.get(g, '')} | {len(idx)} | " + ' | '.join(cells) + ' |')

    L += ['', '## Uji McNemar (gabungan vs tunggal)', '',
          '| Perbandingan | Hanya gabungan benar | Hanya tunggal benar | p-value |',
          '|---|---|---|---|']
    for key, name in SCEN[:2]:
        b, c, p = mcnemar(y, preds['gabungan'], preds[key])
        L.append(f'| Gabungan vs {name} | {b} | {c} | {p:.4f} |')

    with open(os.path.join(out, 'ringkasan.md'), 'w') as f:
        f.write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
