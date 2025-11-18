import io
import base64
import requests
from PIL import Image
import streamlit as st
from huggingface_hub import InferenceClient

# =========================
# CONFIGURAÇÕES GERAIS
# =========================

st.set_page_config(
    page_title="Laura Image Studio",
    page_icon="🖼️",
    layout="centered"
)

st.title("🖼️ Laura Image Studio")

# ---- Senha simples de acesso ----
APP_PASSWORD = st.secrets.get("APP_PASSWORD", "1234")

senha = st.text_input("Senha de acesso", type="password")
if senha != APP_PASSWORD:
    st.warning("Digite a senha correta para acessar o gerador de imagens.")
    st.stop()

st.success("Acesso liberado! ✅")

# ---- Chaves das APIs nos secrets ----
LEMONFOX_API_KEY = st.secrets.get("LEMONFOX_API_KEY", "")
HF_TOKEN = st.secrets.get("HF_TOKEN", "")

# =========================
# ENDPOINTS / MODELOS
# =========================

# LemonFox (SDXL)
LEMONFOX_URL = "https://api.lemonfox.ai/v1/images/generations"

# Hugging Face – FLUX.1-dev via Router
HF_MODEL_FLUX = "black-forest-labs/FLUX.1-dev"
HF_API_BASE_URL = "https://router.huggingface.co/hf-inference/models"  # /<MODEL_ID>

# Hugging Face – Qwen Image via InferenceClient (fal-ai)
QWEN_IMAGE_MODEL = "Qwen/Qwen-Image"

# Hugging Face – Playground v2.5 via InferenceClient (fal-ai)
PLAYGROUND_MODEL = "playgroundai/playground-v2.5-1024px-aesthetic"

# =========================
# PROMPTS DA LAURA – HQ BIQUÍNI
# =========================

PROMPT_LAURA_HQ_BIQUINI = (
    "Laura Massariol, beautiful Brazilian redhead woman in her late 20s, "
    "long wavy fiery red hair, bright green eyes, playful confident smile, "
    "curvy hourglass body, full hips, thick thighs, natural medium-large breasts, "
    "wearing a stylish Brazilian bikini on the beach, sunny late afternoon, "
    "standing in a dynamic sexy pose, one hand on her hip, other hand touching her hair, "
    "highly detailed COMIC BOOK illustration, adult graphic novel style, "
    "bold clean ink lines, rich cel shading, soft halftone textures, "
    "warm saturated colors, dramatic backlighting outlining her silhouette, "
    "strong contrast between light and shadow, slight grain like printed comics, "
    "camera angle slightly from below to emphasize presence and power, "
    "background with simplified beach and sky, depth of field like comics panel, "
    "ultra detailed, sharp, high resolution, cover art of an adult comic book"
)

NEGATIVE_HQ_DEFAULT = (
    "blurry, low quality, pixelated, noisy, "
    "photorealistic, 3d render, cgi, video game graphics, "
    "anime style, manga style, chibi, cartoon for kids, "
    "bad anatomy, deformed body, extra limbs, fused limbs, "
    "flat chest, flat butt, unnatural skinny body, "
    "warped face, asymmetrical eyes, melted eyes, "
    "mutated hands, extra fingers, missing fingers, "
    "distorted proportions, extreme fisheye, "
    "text, watermark, logo, caption, speech bubble, "
    "overexposed, oversaturated neon colors"
)

# =========================
# FUNÇÕES AUXILIARES
# =========================

def baixar_imagem_url(url: str) -> Image.Image:
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


def download_button_from_pil(img: Image.Image, filename: str, label: str):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    st.download_button(
        label=label,
        data=buf.getvalue(),
        file_name=filename,
        mime="image/png",
    )


# =========================
# CHAMADAS ÀS APIS
# =========================

def gerar_imagens_lemonfox(prompt: str, negative_prompt: str, n: int = 1, size: str = "1024x1024"):
    """
    Gera imagens via LemonFox SDXL.
    Retorna uma lista de URLs.
    """
    if not LEMONFOX_API_KEY:
        raise RuntimeError("LEMONFOX_API_KEY não encontrado em st.secrets.")

    headers = {
        "Authorization": f"Bearer {LEMONFOX_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "prompt": prompt,
        "n": n,
        "tamanho": size,
        "formato_de_resposta": "url",
    }

    if negative_prompt:
        payload["prompt_negativo"] = negative_prompt

    resp = requests.post(LEMONFOX_URL, headers=headers, json=payload, timeout=120)

    if resp.status_code != 200:
        st.error(f"Erro da API LemonFox (status {resp.status_code})")
        st.code(resp.text)
        raise RuntimeError(f"Falha LemonFox: {resp.status_code}")

    data = resp.json()
    urls = [item["url"] for item in data.get("data", [])]
    return urls


def gerar_imagens_flux_router(prompt: str, negative_prompt: str, n: int = 1, size: str = "1024x1024"):
    """
    Gera imagens via Hugging Face Router com o modelo FLUX.1-dev.
    Retorna lista de PIL.Image.
    """
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN não encontrado em st.secrets.")

    api_url = f"{HF_API_BASE_URL}/{HF_MODEL_FLUX}"

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Accept": "image/png",
    }

    params = {
        "negative_prompt": negative_prompt or "",
        "num_inference_steps": 26,
        "guidance_scale": 7.0,
    }

    try:
        w, h = size.lower().split("x")
        params["width"] = int(w)
        params["height"] = int(h)
    except Exception:
        pass

    imagens = []

    for i in range(n):
        payload = {
            "inputs": prompt,
            "parameters": params,
        }

        resp = requests.post(api_url, headers=headers, json=payload, timeout=240)
        if resp.status_code != 200:
            st.error(f"Erro da API Hugging Face (FLUX.1-dev) – status {resp.status_code} na imagem {i+1}")
            st.code(resp.text)
            raise RuntimeError(f"Falha HF FLUX: {resp.status_code}")

        try:
            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        except Exception as e:
            st.error(f"Não consegui decodificar a imagem {i+1} do FLUX.1-dev.")
            st.code(str(e))
            raise

        imagens.append(img)

    return imagens


def gerar_imagens_qwen(prompt: str, negative_prompt: str, n: int = 1):
    """
    Gera imagens usando Qwen/Qwen-Image via InferenceClient (provider fal-ai).
    Retorna lista de PIL.Image.
    """
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN não encontrado em st.secrets.")

    client = InferenceClient(
        provider="fal-ai",
        api_key=HF_TOKEN,
    )

    full_prompt = prompt
    if negative_prompt:
        full_prompt = f"{prompt}. Avoid: {negative_prompt}"

    imagens = []
    for i in range(n):
        img = client.text_to_image(
            prompt=full_prompt,
            model=QWEN_IMAGE_MODEL,
        )
        imagens.append(img)

    return imagens


def gerar_imagens_playground(prompt: str, negative_prompt: str, n: int = 1):
    """
    Gera imagens usando playgroundai/playground-v2.5-1024px-aesthetic
    via InferenceClient (provider fal-ai).
    Retorna lista de PIL.Image.
    """
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN não encontrado em st.secrets.")

    client = InferenceClient(
        provider="fal-ai",
        api_key=HF_TOKEN,
    )

    full_prompt = prompt
    if negative_prompt:
        full_prompt = f"{prompt}. Avoid: {negative_prompt}"

    imagens = []
    for i in range(n):
        img = client.text_to_image(
            prompt=full_prompt,
            model=PLAYGROUND_MODEL,
        )
        imagens.append(img)

    return imagens


# =========================
# UI – FORMULÁRIO
# =========================

st.subheader("Configuração do Prompt – Laura HQ de biquíni")

provider = st.radio(
    "Escolha o provedor:",
    [
        "LemonFox (SDXL)",
        "Hugging Face – FLUX.1-dev (Router)",
        "Hugging Face – Qwen-Image (fal-ai)",
        "Hugging Face – Playground v2.5 (fal-ai)",
    ],
    index=1,
)

col1, col2 = st.columns(2)

with col1:
    prompt_positivo = st.text_area(
        "Prompt positivo (Laura em HQ):",
        value=PROMPT_LAURA_HQ_BIQUINI,
        height=200,
    )

with col2:
    prompt_negativo = st.text_area(
        "Prompt negativo (o que evitar):",
        value=NEGATIVE_HQ_DEFAULT,
        height=200,
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
        if provider.startswith("LemonFox"):
            if not LEMONFOX_API_KEY:
                st.error("LEMONFOX_API_KEY não configurada nos secrets.")
                st.stop()

            st.info("Chamando API LemonFox (SDXL)...")
            urls = gerar_imagens_lemonfox(prompt_positivo, prompt_negativo, n=qtd, size=tamanho)

            if not urls:
                st.warning("A LemonFox não retornou URLs de imagens.")
            else:
                for idx, url in enumerate(urls, start=1):
                    st.markdown(f"### Imagem {idx}")
                    try:
                        img = baixar_imagem_url(url)
                        st.image(img, use_column_width=True)
                        download_button_from_pil(img, f"laura_lemonfox_{idx}.png", f"⬇️ Baixar imagem {idx}")
                    except Exception as e:
                        st.error(f"Falha ao baixar a imagem {idx} da LemonFox.")
                        st.code(str(e))

        elif "FLUX.1-dev" in provider:
            if not HF_TOKEN:
                st.error("HF_TOKEN não configurado nos secrets.")
                st.stop()

            st.info(f"Chamando Hugging Face Router – {HF_MODEL_FLUX} ...")
            imagens = gerar_imagens_flux_router(prompt_positivo, prompt_negativo, n=qtd, size=tamanho)

            if not imagens:
                st.warning("A Hugging Face não retornou imagens (FLUX.1-dev).")
            else:
                for idx, img in enumerate(imagens, start=1):
                    st.markdown(f"### Imagem {idx}")
                    st.image(img, use_column_width=True)
                    download_button_from_pil(img, f"laura_flux_{idx}.png", f"⬇️ Baixar imagem {idx}")

        elif "Qwen-Image" in provider:
            if not HF_TOKEN:
                st.error("HF_TOKEN não configurado nos secrets.")
                st.stop()

            st.info(f"Chamando Hugging Face – Qwen/Qwen-Image via fal-ai ...")
            imagens = gerar_imagens_qwen(prompt_positivo, prompt_negativo, n=qtd)

            if not imagens:
                st.warning("Qwen-Image não retornou imagens.")
            else:
                for idx, img in enumerate(imagens, start=1):
                    st.markdown(f"### Imagem {idx}")
                    st.image(img, use_column_width=True)
                    download_button_from_pil(img, f"laura_qwen_{idx}.png", f"⬇️ Baixar imagem {idx}")

        else:  # Playground v2.5
            if not HF_TOKEN:
                st.error("HF_TOKEN não configurado nos secrets.")
                st.stop()

            st.info(f"Chamando Hugging Face – Playground v2.5 via fal-ai ...")
            imagens = gerar_imagens_playground(prompt_positivo, prompt_negativo, n=qtd)

            if not imagens:
                st.warning("Playground v2.5 não retornou imagens.")
            else:
                for idx, img in enumerate(imagens, start=1):
                    st.markdown(f"### Imagem {idx}")
                    st.image(img, use_column_width=True)
                    download_button_from_pil(img, f"laura_playground_{idx}.png", f"⬇️ Baixar imagem {idx}")

    except Exception as e:
        st.error(f"Falha ao gerar imagens: {e}")
