# 大規模言語モデルによる対話アプリ

- お勉強のためのアプリです。
  - さまざまな誘導をWEBUIだけで完結して試したい。
- 参考にしたもの
    - [Google の FLAN-20B with UL2 を動かしてChatGPT APIのように使ってみる！](https://qiita.com/sakasegawa/items/7394fe68eb0087b3c4a5)
        - https://colab.research.google.com/drive/13XP_icIx1Vs6gXjpJlyAJacCwOmQNJzm?usp=sharing
    - [Prompt-Engineering-Guide](https://github.com/dair-ai/Prompt-Engineering-Guide)

## ビルド
```bash
docker build -t llm/chatapp .
```

## 起動
```bash
docker run --rm -d --gpus all -p 8501:8501 -v $PWD:/work -w /work llm/chatapp streamlit run app.py
```

## 利用
- ブラウザで localhost:8501にアクセス

## 対象デバイス/(動作を確認した環境)
 - Ubuntu 20.04 LTS
 - Docker version 20.10.12, build
 - NVIDIA-SMI 470.161.03 / CUDA 11.4
 - RTX 3090 24 GiB
 - CPU-memory : 64 GiB
 - Intel(R) Core(TM) i7-6900K CPU @ 3.20GHz
