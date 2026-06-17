import io
import base64
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

st.title("🖼️ Laura Image Studio")

# ---- Senha simples de acesso ----
APP_PASSWORD = st.secrets.get("APP_PASSWORD", "1234")

senha = st.text_input("Senha de acesso", type="password")

if senha != APP_PASSWORD:
    st.warning("Digite a senha correta para acessar o gerador de imagens.")
    st.stop()

st.success("Acesso liberado! ✅")

# ---- Chave LemonFox nos secrets ----
LEMONFOX_API_KEY = st.secrets.get("LEMONFOX_API_KEY", "")

# =========================
# ENDPOINT / MODELO
# =========================

# LemonFox Stable Diffusion XL
LEMONFOX_URL = "https://api.lemonfox.ai/v1/images/generations"

# =========================
# PROMPTS DA LAURA – HQ BIQUÍNI
# =========================

PROMPT_LAURA_HQ_BIQUINI = (
    "Laura Massariol, beautiful Brazilian redhead woman in her late 20s, "
    "long wavy fiery red hair, bright green eyes, playful confident smile, "
    "curvy hourglass body, full hips, thick thighs, natural medium-large breasts, "
    "wearing a stylish Brazilian bikini on the beach, sunny late afternoon, "
    "standing in a dynamic pose, one hand on her hip, other hand touching her hair, "
    "highly detailed Western comic book illustration, adult graphic novel style, "
    "not anime, not manga, not oriental, "
    "bold clean ink lines, rich cel shading, soft halftone textures, "
    "warm saturated colors, dramatic backlighting outlining her silhouette, "
    "strong contrast between light and shadow, slight grain like printed comics, "
    "background with simplified beach and sky, depth of field like comics panel, "
    "ultra detailed, sharp, high resolution, comic book cover art"
)

NEGATIVE_HQ_DEFAULT = (
    "blurry, low quality, pixelated, noisy, "
    "photorealistic, realistic photo, 3d render, cgi, video game graphics, "
    "anime style, manga style, chibi, cartoon for kids, oriental style, "
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
    """
    Baixa uma imagem a partir de uma URL temporária retornada pela LemonFox.
    Retorna PIL.Image em RGB.
    """
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


def imagem_from_b64_json(b64_json: str) -> Image.Image:
    """
    Converte uma imagem em base64 retornada pela LemonFox para PIL.Image.
    """
    img_bytes = base64.b64decode(b64_json)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


def download_button_from_pil(img: Image.Image, filename: str, label: str):
    """
    Cria botão de download para uma imagem PIL.
    """
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    st.download_button(
        label=label,
        data=buf.getvalue(),
        file_name=filename,
        mime="image/png",
    )


# =========================
# CHAMADA À API LEMONFOX
# =========================

def gerar_imagens_lemonfox(
    prompt: str,
    negative_prompt: str = "",
    n: int = 1,
    size: str = "1024x1024",
    response_format: str = "url",
):
    """
    Gera imagens via LemonFox SDXL.

    Parâmetros:
    - prompt: prompt positivo.
    - negative_prompt: prompt negativo.
    - n: quantidade de imagens.
    - size: tamanho da imagem. Ex: 1024x1024.
    - response_format: "url" ou "b64_json".

    Retorna:
    - lista de URLs, se response_format="url"
    - lista de PIL.Image, se response_format="b64_json"
    """

    if not LEMONFOX_API_KEY:
        raise RuntimeError("LEMONFOX_API_KEY não encontrado em st.secrets.")

    if response_format not in ["url", "b64_json"]:
        raise ValueError("response_format deve ser 'url' ou 'b64_json'.")

    headers = {
        "Authorization": f"Bearer {LEMONFOX_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "prompt": prompt.strip(),
        "n": int(n),
        "tamanho": size,
        "formato_de_resposta": response_format,
    }

    if negative_prompt and negative_prompt.strip():
        payload["prompt_negativo"] = negative_prompt.strip()

    resp = requests.post(
        LEMONFOX_URL,
        headers=headers,
        json=payload,
        timeout=180,
    )

    if resp.status_code != 200:
        st.error(f"Erro da API LemonFox — status {resp.status_code}")
        st.code(resp.text)
        raise RuntimeError(f"Falha LemonFox: {resp.status_code}")

    data = resp.json()
    itens = data.get("data", [])

    if not itens:
        return []

    if response_format == "url":
        urls = []

        for item in itens:
            url = item.get("url")
            if url:
                urls.append(url)

        return urls

    imagens = []

    for item in itens:
        b64_json = item.get("b64_json")
        if not b64_json:
            continue

        img = imagem_from_b64_json(b64_json)
        imagens.append(img)

    return imagens


# =========================
# UI – FORMULÁRIO
# =========================

st.subheader("Configuração do Prompt – Laura HQ de biquíni")

col1, col2 = st.columns(2)

with col1:
    prompt_positivo = st.text_area(
        "Prompt positivo:",
        value=PROMPT_LAURA_HQ_BIQUINI,
        height=230,
    )

with col2:
    prompt_negativo = st.text_area(
        "Prompt negativo:",
        value=NEGATIVE_HQ_DEFAULT,
        height=230,
    )

col_a, col_b, col_c = st.columns(3)

with col_a:
    qtd = st.slider(
        "Quantidade de imagens",
        min_value=1,
        max_value=4,
        value=2,
    )

with col_b:
    tamanho = st.selectbox(
        "Tamanho",
        [
            "1024x1024",
            "768x1024",
            "1024x768",
        ],
        index=0,
        help="Tamanho enviado para a API LemonFox SDXL.",
    )

with col_c:
    formato_resposta = st.selectbox(
        "Formato da resposta",
        [
            "url",
            "b64_json",
        ],
        index=0,
        help=(
            "url: retorna links temporários das imagens. "
            "b64_json: retorna a imagem diretamente em base64."
        ),
    )

st.divider()

# =========================
# BOTÃO DE GERAÇÃO
# =========================

if st.button("🚀 Gerar imagens da Laura"):
    if not prompt_positivo.strip():
        st.error("Digite um prompt positivo.")
        st.stop()

    if not LEMONFOX_API_KEY:
        st.error("LEMONFOX_API_KEY não configurada nos secrets.")
        st.stop()

    try:
        st.info("Chamando API LemonFox SDXL...")

        resultado = gerar_imagens_lemonfox(
            prompt=prompt_positivo,
            negative_prompt=prompt_negativo,
            n=qtd,
            size=tamanho,
            response_format=formato_resposta,
        )

        if not resultado:
            st.warning("A LemonFox não retornou imagens.")
            st.stop()

        st.success("Imagem(ns) gerada(s) com sucesso! ✅")

        if formato_resposta == "url":
            for idx, url in enumerate(resultado, start=1):
                st.markdown(f"### Imagem {idx}")

                try:
                    img = baixar_imagem_url(url)

                    st.image(
                        img,
                        use_column_width=True,
                    )

                    st.caption(
                        "URL temporária da LemonFox. "
                        "Normalmente fica disponível por tempo limitado."
                    )

                    with st.expander("Ver URL da imagem"):
                        st.code(url)

                    download_button_from_pil(
                        img,
                        f"laura_lemonfox_{idx}.png",
                        f"⬇️ Baixar imagem {idx}",
                    )

                except Exception as e:
                    st.error(f"Falha ao baixar a imagem {idx}.")
                    st.code(str(e))

        else:
            for idx, img in enumerate(resultado, start=1):
                st.markdown(f"### Imagem {idx}")

                st.image(
                    img,
                    use_column_width=True,
                )

                download_button_from_pil(
                    img,
                    f"laura_lemonfox_{idx}.png",
                    f"⬇️ Baixar imagem {idx}",
                )

    except Exception as e:
        st.error(f"Falha ao gerar imagens: {e}")
