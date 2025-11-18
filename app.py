import os
import base64
import io
import requests
from PIL import Image
import streamlit as st

# =========================
# CONFIGURAÇÕES GERAIS
# =========================

st.set_page_config(
    page_title="Laura Image Studio",
    page_icon="🖼️",
    layout="centered"
)

# ---- Senha simples de acesso ----
APP_PASSWORD = st.secrets.get("APP_PASSWORD", "1234")

st.title("🖼️ Laura Image Studio")

senha = st.text_input("Senha de acesso", type="password")
if senha != APP_PASSWORD:
    st.warning("Digite a senha correta para acessar o gerador de imagens.")
    st.stop()

st.success("Acesso liberado!")

# ---- Chaves das APIs nos secrets ----
LEMONFOX_API_KEY = st.secrets.get("LEMONFOX_API_KEY", "")
HF_TOKEN = st.secrets.get("HF_TOKEN", "")

# =========================
# MODELOS / CONSTANTES
# =========================

# LemonFox (SDXL)
LEMONFOX_URL = "https://api.lemonfox.ai/v1/images/generations"

# Hugging Face Inference – agora com endpoint CORRETO
HF_MODEL_ID = "black-forest-labs/FLUX.1-dev"
HF_API_BASE_URL = "https://router.huggingface.co/hf-inference/models"  # base
# URL final será: f"{HF_API_BASE_URL}/{HF_MODEL_ID}"


# =========================
# PROMPTS DA LAURA
# =========================

PROMPT_LAURA_BIQUINI = (
    "beautiful young woman named Laura, 25 years old, redhead ponytail, "
    "curvy body, full hips and round butt, medium big breasts, "
    "wearing a tiny Brazilian bikini on the beach, confident pose, "
    "warm sunlight, dramatic shadows, highly detailed, 8k, "
    "photorealistic, cinematic lighting"
)

NEGATIVE_LAURA_DEFAULT = (
    "ugly, deformed, extra limbs, bad anatomy, lowres, blurry, "
    "text, watermark, logo, disfigured face, mutated hands, "
    "cartoon, anime, sketch, low quality"
)


# =========================
# FUNÇÕES AUXILIARES
# =========================

def baixar_imagem(url: str) -> Image.Image:
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


def exibir_download(imagem: Image.Image, nome_arquivo: str):
    buf = io.BytesIO()
    imagem.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    href = f'<a download="{nome_arquivo}" href="data:file/png;base64,{b64}">⬇️ Baixar imagem</a>'
    st.markdown(href, unsafe_allow_html=True)


# =========================
# CHAMADAS ÀS APIS
# =========================

def gerar_imagens_lemonfox(prompt: str, negative_prompt: str, n: int = 1, size: str = "1024x1024"):
    if not LEMONFOX_API_KEY:
        raise RuntimeError("LEMONFOX_API_KEY não encontrado em st.secrets.")

    headers = {
        "Authorization": f"Bearer {LEMONFOX_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt or None,
        "n": n,
        "size": size,
        "response_format": "url",
    }

    resp = requests.post(LEMONFOX_URL, headers=headers, json=payload, timeout=120)

    if resp.status_code != 200:
        st.error(f"Erro da API LemonFox (status {resp.status_code})")
        st.code(resp.text)
        raise RuntimeError(f"Falha LemonFox: {resp.status_code}")

    data = resp.json()
    urls = [item["url"] for item in data.get("data", [])]
    return urls


def gerar_imagens_hf(prompt: str, negative_prompt: str, n: int = 1, size: str = "1024x1024"):
    """
    Geração via Hugging Face Inference (router.huggingface.co).

    IMPORTANTE:
    - Endpoint correto: https://router.huggingface.co/hf-inference/models/<MODEL_ID>
    - Corpo: {"inputs": "...", "parameters": {...}}
    - Retorno: bytes de imagem (PNG/JPEG).
    """

    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN não encontrado em st.secrets.")

    # Monta a URL COM o modelo no path (era isso que estava dando 404)
    api_url = f"{HF_API_BASE_URL}/{HF_MODEL_ID}"

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Accept": "image/png",  # pedimos imagem direta
    }

    # Parâmetros básicos (muitos modelos ignoram alguns campos,
    # mas negativo, steps e tamanho costumam ser respeitados)
    params = {
        "negative_prompt": negative_prompt or "",
        "num_inference_steps": 28,
        "guidance_scale": 7.0,
    }

    # Tenta interpretar largura x altura
    try:
        w, h = size.lower().split("x")
        params["width"] = int(w)
        params["height"] = int(h)
    except Exception:
        pass  # se der erro, deixa o modelo escolher o tamanho padrão

    imagens = []

    for i in range(n):
        payload = {
            "inputs": prompt,
            "parameters": params,
        }

        resp = requests.post(api_url, headers=headers, json=payload, timeout=240)

        if resp.status_code != 200:
            st.error(f"Erro da API Hugging Face (status {resp.status_code}) na imagem {i+1}")
            # Mostra texto bruto pra debug (é aqui que você veria 404, 410, etc.)
            st.code(resp.text)
            raise RuntimeError(f"Falha HF: {resp.status_code}")

        try:
            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        except Exception as e:
            st.error(f"Não consegui decodificar a imagem {i+1} da HF.")
            st.code(str(e))
            raise

        imagens.append(img)

    return imagens


# =========================
# UI – FORMULÁRIO
# =========================

st.subheader("Configuração do Prompt")

modelo = st.radio(
    "Qual provedor usar?",
    ["LemonFox (SDXL)", "Hugging Face (FLUX.1-dev)"],
    index=0,
    help="Se o LemonFox estiver com erro 500, teste o Hugging Face."
)

col1, col2 = st.columns(2)

with col1:
    prompt_positivo = st.text_area(
        "Prompt positivo (Laura)",
        value=PROMPT_LAURA_BIQUINI,
        height=180,
    )

with col2:
    prompt_negativo = st.text_area(
        "Prompt negativo",
        value=NEGATIVE_LAURA_DEFAULT,
        height=180,
    )

col_a, col_b = st.columns(2)
with col_a:
    qtd = st.slider("Quantidade de imagens", 1, 4, 2)
with col_b:
    tamanho = st.selectbox(
        "Tamanho",
        ["1024x1024", "768x1024", "1024x768"],
        index=0,
    )

if st.button("🚀 Gerar imagens da Laura"):
    if not prompt_positivo.strip():
        st.error("Digite um prompt positivo.")
        st.stop()

    try:
        if modelo.startswith("LemonFox"):
            st.info("Chamando API LemonFox (SDXL)...")
            urls = gerar_imagens_lemonfox(prompt_positivo, prompt_negativo, n=qtd, size=tamanho)
            if not urls:
                st.warning("A LemonFox não retornou URLs de imagens.")
            else:
                for idx, url in enumerate(urls, start=1):
                    st.markdown(f"### Imagem {idx}")
                    try:
                        img = baixar_imagem(url)
                        st.image(img, use_column_width=True)
                        exibir_download(img, f"laura_lemonfox_{idx}.png")
                    except Exception as e:
                        st.error(f"Falha ao baixar a imagem {idx} da LemonFox.")
                        st.code(str(e))

        else:
            st.info(f"Chamando Hugging Face ({HF_MODEL_ID}) via HF Inference...")
            imagens = gerar_imagens_hf(prompt_positivo, prompt_negativo, n=qtd, size=tamanho)
            if not imagens:
                st.warning("A Hugging Face não retornou imagens.")
            else:
                for idx, img in enumerate(imagens, start=1):
                    st.markdown(f"### Imagem {idx}")
                    st.image(img, use_column_width=True)
                    exibir_download(img, f"laura_hf_{idx}.png")

    except Exception as e:
        st.error(f"Falha ao gerar imagens: {e}")
