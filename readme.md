# Lab3 Multi-class Classification with Chest X-Ray (CXR) Dataset

本專案為 **Lab3 Kaggle Multi-class Classification** 競賽之完整深度學習訓練與推論系統，針對胸部 X 光（Chest X-Ray, CXR）影像進行四類別互斥診斷分類（`normal`、`bacteria`、`virus`、`COVID-19`），並針對競賽評分指標 **Macro-F1** 進行端到端最佳化。

## 1. 專案特色與核心架構

本系統將所有前處理、跨折交叉驗證、異質模型集成與後處理校準完整封裝於 `main.py`，具備以下核心設計：

1. **三通道雙尺度 CLAHE 醫學影像增強 (`CLAHETransform`)**：
   將單通道灰階 X 光片轉換為 `[Raw Grayscale, Standard CLAHE (clip=2.0, grid=8×8), High-Contrast CLAHE (clip=3.5, grid=12×12)]` 三通道輸入，兼顧整體胸腔對比與局部毛玻璃狀病灶（GGO）/ 肺實質浸潤細節。
2. **三大異質骨幹網路 × 5-Fold 分層交叉驗證集成 (15-Model Heterogeneous Ensemble)**：
   合併 `train_data.csv`（4,017 張）與 `val_data.csv`（709 張）共 4,726 張影像，使用 `StratifiedKFold(n_splits=5)` 依序訓練：
   * `convnext_small`（$7 \times 7$ Depthwise Conv 全域感受野）
   * `densenet121`（CheXNet 經典跨層密集特徵重用）
   * `efficientnet_v2_s`（高解析度細粒度局部紋理擷取）
3. **GeM 空間池化 + Multi-Sample Dropout + Model EMA**：
   採用可學習廣義平均池化（Generalized Mean Pooling, $p=3.0$）聚焦局部病灶，搭配 4 分支平行 Multi-Sample Dropout（$p \in \{0.2, 0.3, 0.4, 0.5\}$）與模型指數移動平均（EMA, $\text{decay}=0.995$），大幅提升少數類別泛化能力。
4. **極端類別不平衡處理 (`LogitAdjustedFocalLoss` + 阻尼抽樣)**：
   針對僅佔 $0.97\%$ 的 `COVID-19` 類別，結合阻尼加權隨機抽樣（$w_c \propto N_c^{-0.50}$）、有效樣本數加權（Effective Number, $\beta=0.999$）、Focal Loss（$\gamma=2.0$）、Label Smoothing（$\varepsilon=0.04$）與批次級 `Mixup`（$\alpha=0.2$）。
5. **4 視角測試期擴增 (4x TTA) 與雙重後處理校準**：
   推論時結合原圖、水平翻轉、中心 $92\%$ 微縮放與微縮放翻轉 4 個視角，並透過 **Nelder-Mead OOF 機率乘數最佳化** 與 **Spec 第 6 頁測試集先驗分佈對齊（Prior Alignment: `315 normal, 556 bacteria, 299 virus, 12 COVID-19`）** 直接最大化 Macro-F1。

## 2. 資料夾結構

請確保資料集檔案與 `main.py` 放置於同一目錄（預設路徑為 `C:\Users\lueri\Desktop\Lab3_Kaggle`，程式亦支援自動偵測當前所在目錄，並自動遞迴建立所有影像路徑索引）：

```text
Lab3_Kaggle/
├── main.py                      # 免參數一鍵執行之完整訓練、集成推論與圖表生成主程式
├── README.md                    # 專案說明文件
├── train_data.csv               # 訓練集標籤 (4,017 筆)
├── val_data.csv                 # 驗證集標籤 (709 筆)
├── test_data_sample.csv         # 測試集提交範本與檔名順序 (1,182 筆)
├── train_images/                # 訓練集 X 光影像資料夾 (自動遞迴搜尋)
├── val_images/                  # 驗證集 X 光影像資料夾 (自動遞迴搜尋)
├── test_images/                 # 測試集 X 光影像資料夾 (自動遞迴搜尋)
├── submission.csv               # 執行完成後自動生成之最終最佳 Kaggle 提交檔
└── outputs/                     # 執行完成後自動建立之輸出資料夾
    ├── checkpoints/             # 各骨幹網路與各 Fold 最佳 EMA 模型權重 (.pth) 與 OOF 快取 (.npy)
    ├── submission_best_prior_aligned.csv  # 對齊測試集先驗分佈之最佳提交檔
    ├── submission_oof_calibrated.csv      # 純 OOF 機率乘數校準之提交檔
    ├── ablation_comparison.csv            # 實驗報告用之各模型與策略 OOF Macro-F1 比較表
    ├── training_curves.png                # 訓練與驗證 Loss / Macro-F1 收斂曲線圖 (300 DPI)
    ├── confusion_matrix.png               # 異質集成模型 OOF 四分類混淆矩陣 (300 DPI)
    ├── class_f1_breakdown.png             # 各類別 F1-Score 與整體 Macro-F1 長條圖 (300 DPI)
    └── model_ensemble_comparison.png      # 骨幹網路與最佳化策略消融比較圖 (300 DPI)
```

## 3. 環境需求與安裝

建議使用 Python 3.9+ 與具備 CUDA 支援之 PyTorch 環境執行：

```bash
pip install torch torchvision scikit-learn opencv-python scipy pandas pillow matplotlib seaborn
```

## 4. 執行與訓練方式

所有超參數與檔案路徑皆已直接寫入 `main.py` 頂部的 `CONFIG` 字典中，無須輸入任何指令列參數，直接執行即可：

```bash
cd /d "C:\Users\lueri\Desktop\Lab3_Kaggle"
python main.py
```

### 斷點續訓機制 (Auto-Resume)

`main.py` 內建自動斷點續訓功能（`RESUME_CHECKPOINTS: True`）。每當一個 Fold 訓練完成，最佳模型權重與驗證預測機率會立即儲存於 `outputs/checkpoints/`。若訓練中途因故中斷，再次執行 `python main.py` 時會自動載入已完成的 Fold 並接續訓練剩餘部分。

## 5. 自訂設定說明（選用）

若需根據硬體環境微調，可直接修改 `main.py` 頂部的 `CONFIG` 字典：

| `CONFIG` 鍵值 | 預設值 | 說明 |
| :--- | :--- | :--- |
| `DATA_DIR` | `r"C:\Users\lueri\Desktop\Lab3_Kaggle"` | 資料集根目錄路徑（若找不到會自動退回 `main.py` 所在目錄） |
| `ARCHITECTURES` | `["convnext_small", "densenet121", "efficientnet_v2_s"]` | 參與異質集成的骨幹網路清單 |
| `NUM_FOLDS` | `5` | Stratified K-Fold 折數 |
| `IMG_SIZE` | `384` | 輸入影像解析度（若 GPU VRAM 小於 8GB 可改為 `320`） |
| `BATCH_SIZE` | `16` | 單步批次大小（搭配 `GRAD_ACCUM_STEPS: 2` 等效批次大小為 `32`） |
| `EPOCHS` | `22` | 每個 Fold 的訓練週期數（含前 2 個 Epoch 線性 Warmup） |
