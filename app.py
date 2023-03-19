"""Copyright 2023 @gyumaruya

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
import logging
import os
import shutil

import streamlit as st
import torch
from accelerate import (
    infer_auto_device_map,
    init_empty_weights,
    load_checkpoint_and_dispatch,
)
from huggingface_hub import snapshot_download
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    T5ForConditionalGeneration,
    T5Tokenizer,
    pipeline,
)

logger = logging.getLogger(__name__)

# --------------------
#   INITIALIZE UTIL
# --------------------
def _mv_git_snapshot(src, dst):
    logger.debug(f"move folder : {src} to {dst}")
    # .cacheにダウンロードされたgit-snapshotをシンボリックリンク解決してコピーする
    os.makedirs(dst, exist_ok=True)
    for p in os.listdir(src):
        logger.debug(f"fix file : {os.path.join(src, p)} to {os.path.join(dst, p)}")
        shutil.copy(os.path.join(src, p), os.path.join(dst, p), follow_symlinks=True)


def _cache_load_model(name):
    # あればローカルを利用。ないときはダウンロード.
    model_dir = os.path.join("/work", name)
    if not os.path.exists(model_dir):
        logger.debug("download model.")
        download_path = snapshot_download(repo_id=name)
        logger.debug("finish download model.")

        logger.debug("move file")
        _mv_git_snapshot(download_path, model_dir)
    return model_dir


# ------------------------
#    MODEL CONFIG
# ------------------------
# 高速化のためキャッシュ済み。
@st.cache_resource()
def load_model(modelname):
    logger.debug("start load model.")
    new_path = _cache_load_model(modelname)

    logger.debug("start load tokenizer.")
    tokenizer = T5Tokenizer.from_pretrained(new_path)
    # ここで日本語モデルを入れてもfine-tuneをしていないので、トークンIDがズレるため不可能
    # tokenizer = T5Tokenizer.from_pretrained("sonoisa/t5-base-japanese")
    logger.debug("tokernizer loaded")

    # GPUに全部は乗らないので、CPUメモリに載せる。
    # RTX 3090想定
    max_memory_mapping = {0: "22GiB", "cpu": "60GiB"}

    if modelname == "google/flan-ul2":
        # 8bitにしてもGPUに乗らないので、float16でCPUにも乗せる。
        # 一度CPUに読み込ませたものを再割り当てするのが肝
        config = AutoConfig.from_pretrained(new_path)
        with init_empty_weights():
            model = AutoModelForSeq2SeqLM.from_config(config)
        device_map = infer_auto_device_map(
            model,
            max_memory=max_memory_mapping,
            dtype=torch.float16,
            no_split_module_classes=["T5Blocks"],
        )
        model = load_checkpoint_and_dispatch(
            model, dtype=torch.float16, checkpoint=new_path, device_map=device_map
        ).eval()
    else:
        # 8bit量子化allGPU
        model_8bit = T5ForConditionalGeneration.from_pretrained(
            new_path,
            max_memory=max_memory_mapping,
            load_in_8bit=True,
            device_map="auto",
        )
        model = model_8bit
    logger.debug("model loaded")
    return tokenizer, model


@st.cache_resource()
def load_translater(trans_en2ja, trans_ja2en):
    # ほんやくコンニャク
    logger.debug("download translater")
    trans_en2ja_path = _cache_load_model(trans_en2ja)
    trans_ja2en_path = _cache_load_model(trans_ja2en)

    # load model
    trans_en2ja = pipeline("translation", model=trans_en2ja_path, device=-1)
    trans_ja2en = pipeline("translation", model=trans_ja2en_path, device=-1)

    return trans_en2ja, trans_ja2en


# ここでChat Botのコンテキストになりきってもらう。
# これがzero-shot-learning-CoT
# > https://github.com/dair-ai/Prompt-Engineering-Guide/blob/main/guides/prompts-basic-usage.md
prefix_text = """## Self-Introduction
Below is the conversation between the {role_name_human} and the {role_name_ai}. The secretary's tone is technical and scientific.

{role_name_human}: Hello, who are you?
{role_name_ai}: Greetings! I'm a secretary. Is there anything I can help you with today?
{role_name_human}: Tell me about the creation of black holes.
{role_name_ai}: I see! A black hole is a region of spacetime whose gravity is so strong that not even light can escape it. They are created when a very massive star dies and its core collapses, forming a singularity of infinite density. The singularity's intense gravity pulls all matter and radiation around it, creating a black hole.
"""
suffix_text = ""

# ----------------------
#   SYSTEM AUTO CONFIG
# ----------------------
# 会話履歴を記録する
if "message_history" not in st.session_state:
    logger.debug("initialize streamlit")
    st.session_state["message_history"] = []

# ------------------------
#  ESTIMATE PERFORMANCE
# ------------------------
if "token_max_length" not in st.session_state:
    st.session_state["token_max_length"] = 2048

if "token_bos_token_id" not in st.session_state:
    st.session_state["token_bos_token_id"] = 0

if "token_max_time" not in st.session_state:
    st.session_state["token_max_time"] = 30

if "token_min_new_tokens" not in st.session_state:
    st.session_state["token_min_new_tokens"] = 20

if "prefix_text" not in st.session_state:
    st.session_state["prefix_text"] = prefix_text

if "role_name_ai" not in st.session_state:
    st.session_state["role_name_ai"] = "Secretary"
if "role_name_human" not in st.session_state:
    st.session_state["role_name_human"] = "Researcher"

if "TRANS_ENABLE" not in st.session_state:
    st.session_state["TRANS_ENABLE"] = True

# --------------------
#   MODEL SELECTION
# --------------------

if "LLM_MODEL" not in st.session_state:
    st.session_state["LLM_MODEL"] = "google/flan-ul2"  # apache-2.0, 20B
    # st.session_state['LLM_MODEL']  = "google/flan-t5-xxl" # apache-2.0 , 11B

if "TRANS_MODEL_JA2EN" not in st.session_state:
    # 2023/03 現在No.1人気モデル(huggingface)
    # huggingfaceへのログインが必要?
    # st.session_state['TRANS_MODEL_JA2EN']  = "Helsinki-NLP/opus-tatoeba-en-ja"  # apache-2.0
    # st.session_state['TRANS_MODEL_EN2JA']  = "Helsinki-NLP/opus-tatoeba-ja-en"  #  apache-2.0

    # 2023/03 現在No.2人気モデル(huggingface)
    st.session_state["TRANS_MODEL_JA2EN"] = "staka/fugumt-ja-en"  # cc-by-sa-4.0
    st.session_state["TRANS_MODEL_EN2JA"] = "staka/fugumt-en-ja"  # cc-by-sa-4.0


def main():
    st.markdown(
        f"""# 大規模言語モデルによる対話アプリ

- 現在のモデル : {st.session_state['LLM_MODEL']}
- 現在の翻訳モデル(ja -> en): {st.session_state['TRANS_MODEL_JA2EN']}
- 現在の翻訳モデル(en -> ja): {st.session_state['TRANS_MODEL_EN2JA']}
- 現在の翻訳モードは {st.session_state["TRANS_ENABLE"]} です。

推論などの設定はサイドバーから設定出来ます。  
日本語への変換はBERTベースの翻訳モデルを利用します。そのため正確性に欠けます。  
    - CPUのみで十分に動きます。めっちゃ早い。小さい。すごい。
 """
    )

    # メモリ節約
    with torch.no_grad():

        # 時間がかかるので誤魔化す。
        with st.spinner(text="Initializing ...(~5 min)"):
            tokenizer, model = load_model(st.session_state["LLM_MODEL"])
            trans_en2ja, trans_ja2en = load_translater(
                st.session_state["TRANS_MODEL_EN2JA"],
                st.session_state["TRANS_MODEL_JA2EN"],
            )
            logger.debug(model.device_map)

        with st.sidebar:
            # ユーザに動的設定を許したかったらここを変更する。
            st.session_state["token_max_length"] = st.number_input(
                "token_max_length",
                min_value=0,
                max_value=2048,
                value=st.session_state["token_max_length"],
                step=1,
                key="n_token_max_length",
            )
            st.session_state["token_bos_token_id"] = st.number_input(
                "token_bos_token_id",
                min_value=0,
                max_value=2048,
                value=st.session_state["token_bos_token_id"],
                step=1,
                key="n_token_bos_token_id",
            )
            st.session_state["token_max_time"] = st.number_input(
                "token_max_time",
                min_value=1,
                max_value=3600,
                value=st.session_state["token_max_time"],
                step=1,
                key="n_token_max_time",
            )
            st.session_state["token_min_new_tokens"] = st.number_input(
                "token_min_new_tokens",
                min_value=0,
                max_value=2048,
                value=st.session_state["token_min_new_tokens"],
                step=1,
                key="n_token_min_new_tokens",
            )

            st.session_state["TRANS_ENABLE"] = st.checkbox(
                "日本語翻訳", value=st.session_state["TRANS_ENABLE"], key="C_TRANS_ENABLE"
            )

            st.session_state["prefix_text"] = st.text_area(
                "LLMに入れる誘導テキスト",
                value=st.session_state["prefix_text"],
                key="custom_prefix_text",
            )
            st.session_state["role_name_ai"] = st.text_input(
                "誘導テキストないのロール名(role_name_ai と置換される)",
                value=st.session_state["role_name_ai"],
                key="custom_rolen_ame_ai",
            )
            st.session_state["role_name_human"] = st.text_input(
                "誘導テキストないのロール名(role_name_human と置換される)",
                value=st.session_state["role_name_human"],
                key="custom_role_name_human",
            )

            st.markdown(
                """### (参考)モデルの情報
正確性に欠ける可能性があります。参考として下さい。

- OpenAI/chatGPT(gpt3.5-turbo)
  - 4,096 tokens
  - 175B
- Google/Flan-U-PaLM
  - 2,048
  - ??B
- Google/Flan-UL2
  - 2,048 tokens
  - 20B
- Google/Flan-t5-xxl
  - 11B
  - 512? token"""
            )

        # 画面の左側を入力/右側を履歴とする。表示割合を変えたいときにここを変更する。
        action_area, history_area = st.columns([3, 1])

        # 履歴の表示。
        with history_area:
            st.markdown("## 会話履歴: ")
            st.write("十分なパラメータがある時、推論に影響します。")
            st.write("- - -")
            for message in st.session_state["message_history"]:
                # 実際に入力される英語で表記。日本語で表示したい時にここを変える。
                st.write(f"{message['role']}: {message['content']}")

        with action_area:
            input_text = st.text_area("## 入力", key="input")

            # run it !
            if st.button("LLMに推論させる"):
                with st.spinner(text="Estimate Now..."):

                    # 日本語翻訳の有効化
                    if st.session_state["TRANS_ENABLE"]:
                        trans_ja2en_fn = trans_ja2en
                        trans_en2ja_fn = trans_en2ja
                    else:
                        trans_en2ja_fn = lambda x: [{"translation_text": x}]
                        trans_ja2en_fn = lambda x: [{"translation_text": x}]

                    # CPUで翻訳処理。ここではあまり重たい処理は走らない。
                    input_text_en = trans_ja2en_fn(input_text)[0]["translation_text"]

                    # 直近の入力をロールとともに英語で保存.
                    st.session_state["message_history"].append(
                        {
                            "role": st.session_state["role_name_human"],
                            "content": input_text_en,
                        }
                    )

                    # 推論のための文章を組み立て
                    main_text = "".join(
                        [
                            f"{message['role']}: {message['content']}\n"
                            for message in st.session_state["message_history"]
                        ]
                    )
                    _prefix_text = prefix_text.format(
                        role_name_human=st.session_state["role_name_human"],
                        role_name_ai=st.session_state["role_name_ai"],
                    )
                    inputs = _prefix_text + main_text + suffix_text
                    logger.debug("inputs:", inputs)

                    # token化
                    input_ids = tokenizer(inputs, return_tensors="pt").input_ids.to(
                        "cuda"
                    )

                    # 実際の推論
                    outputs = model.generate(
                        input_ids,
                        max_length=st.session_state["token_max_length"],
                        bos_token_id=st.session_state["token_bos_token_id"],
                        max_time=st.session_state["token_max_time"],
                        min_new_tokens=st.session_state["token_min_new_tokens"],
                    )

                    # tokenを英語に変換
                    result = (
                        tokenizer.decode(outputs[0], skip_special_tokens=False)
                        .replace("<pad>", "")
                        .replace("</s>", "")
                        .replace(st.session_state["role_name_ai"] + ":", "")
                        .strip()
                    )

                    # 次回以降のコンテキストのために保存
                    st.session_state["message_history"].append(
                        {"role": st.session_state["role_name_ai"], "content": result}
                    )

                    # CPUで日本語翻訳,そんなに遅くない
                    result = trans_en2ja_fn(result)[0]["translation_text"]

                    logger.debug("outputs", result)

                st.write("## 返答")
                st.write(result)


if __name__ == "__main__":
    main()
