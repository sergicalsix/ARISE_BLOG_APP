import gradio as gr
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai
import os

API_KEY = os.environ['GOOGLE_API_KEY']


lm_name = 'paraphrase-MiniLM-L6-v2'
llm_name ="gemini-pro"
genai.configure(api_key=API_KEY)

# モデル定義
lm = SentenceTransformer(lm_name)
llm= genai.GenerativeModel(llm_name)

# 静的データの読み込み
df = pd.read_csv('data.csv')
urls = df['url'].values
titles = df['title'].values
articles = df['article'].values

# 静的データのembedding(今回は簡単のためタイトルをembedding)
title_embeddings = lm.encode(df['title'], convert_to_tensor=True)


def output(input_text):
    output_text = ""

    # 入力テキストのembedding
    input_embedding = lm.encode(input_text, convert_to_tensor=True)

    # 類似度計算
    similarities = util.pytorch_cos_sim(input_embedding, title_embeddings)

    top3_scores, top3_indices = torch.topk(similarities, k=3)

    top3_scores = top3_scores.cpu().numpy()[0]
    top3_indices = top3_indices.cpu().numpy()[0]

    output_text += f"類似度:{top3_scores}"

    for i in range(len(top3_indices)):
        idx = top3_indices[i]
        score = top3_scores[i]
        
        # 閾値を設定できる。
        if score > 0:
            output_text += f'\n\n############\n#{i+1}番目の記事####\n############'
            output_text += f'\n\n##タイトル: {titles[idx]}'
            chat = llm.start_chat()
            response = chat.send_message(f'以下の入力で示す記事の内容をわかりやすく要約してください。 \n 入力: {articles[idx]}')
            output_text += f'\n\n##要約: {response.text}'
            del chat 
            output_text += f'\n\n##記事URL: {urls[idx]}\n'
        else:
            output_text += f'該当する記事がありませんでした。'
            break 
    return output_text

iface = gr.Interface(fn=output, inputs=gr.components.Textbox(label="検索したいキーワードを入力してください\n参考: https://www.ariseanalytics.com/activities/report/category/arise-tech-blog/"), 
                    outputs=gr.components.Textbox(label="キーワードに合った記事のタイトル、要約、URLを出力します"), title="ARISE テックブログ検索アプリ")
iface.launch(share=True)
