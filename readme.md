# 🌿 PlantHealthMonitor

An image-based plant disease classifier built with **PyTorch** and a Vision Transformer (`vit_b_16`).
Modular, config-driven, Colab-friendly, and designed for reproducible experiments and quick iteration.

---

## ✨ Highlights

* Model: `vit_b_16` (Vision Transformer), supports transfer learning
* Config-driven: experiment settings in `configs/train.yaml`
* Logging & checkpoints: automatic saving to `outputs/logs/` and `outputs/checkpoints/`
* Visuals: training `loss` & `accuracy` plotted and saved after each run
* Colab-ready: simple to run in Google Colab or locally


---

## 📈 Latest training plot

Make sure you have the latest plot copied to a stable filename so it renders in the README. Run from the repo root after training:

```bash
mkdir -p outputs/logs
cp "$(ls -t outputs/logs/report_*.png | head -n1)" outputs/logs/report_latest.png
```

Then include that image in the README (already referenced below). If you prefer a timestamped file, update the image path accordingly.

![Training Loss & Accuracy](/Users/user/Downloads/PlantHealthMonitor/main/Unknown.png)

---

## ⚙️ Quick start (Colab or local)

1. Install requirements:

   ```bash
   pip install -r requirements.txt
   # optional helper
   pip install torch-snippets
   ```

2. (Colab) Mount Drive or clone repo and `cd` into it:

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   %cd /content/PlantHealthMonitor/main
   ```

3. Edit `configs/train.yaml` if needed (paths, epochs, lr, batch size).

4. Run training:

   * Notebook: open `train.ipynb` and run cells.
   * Script:

     ```bash
     python main/train.py
     ```

5. After training, make the latest plot visible:

   ```bash
   cp "$(ls -t outputs/logs/report_*.png | head -n1)" outputs/logs/report_latest.png
   git add outputs/logs/report_latest.png && git commit -m "Add latest training plot" && git push
   ```

---

## 🧾 What’s saved where

* `outputs/logs/train_<timestamp>.log` — plain text epoch metrics
* `outputs/logs/report_<timestamp>.png` — training & validation curves
* `outputs/logs/report_latest.png` — (optional) symlink/copy of the latest plot used in README
* `outputs/checkpoints/epoch_<n>.pt` — per-epoch checkpoints
* `outputs/checkpoints/best_model.pt` — best model by validation accuracy

---

## 🔭 Next steps / suggestions

* Add an inference demo (Streamlit / Gradio) for quick human testing.
* Add Grad-CAM visualizations to explain predictions.
* Track experiments with Weights & Biases for richer analysis.
* Add `scripts/eval.py` to compute per-class metrics and confusion matrices.

---

## 👤 Author

**Big Mann** — Aspiring AI researcher.
Built for fast iteration, solid experiments, and useful results.

---

## 📜 License

MIT © 2025 Big Mann
