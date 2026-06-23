import io
import re
import base64
import requests
from PIL import Image
import streamlit as st

# =========================
# CONFIGURAÇÕES GERAIS
# =========================

st.set_page_config(
    page_title="Comic Book Image Studio",
    page_icon="🖼️",
    layout="centered"
)

st.title("🖼️ Comic Book Image Studio")
st.caption("Transforme uma imagem em estilo comic book usando OpenRouter")

# ---- Senha simples de acesso ----
APP_PASSWORD = st.secrets.get("APP_PASSWORD", "1234")

senha = st.text_input("Senha de acesso", type="password")

if senha != APP_PASSWORD:
    st.warning("Digite a senha correta para acessar o gerador de imagens.")
    st.stop()

st.success("Acesso liberado! ✅")

# ---- Chave OpenRouter nos secrets ----
OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "")
APP_REFERER = st.secrets.get("APP_REFERER", "https://streamlit.app")
APP_TITLE = st.secrets.get("APP_TITLE", "Comic Book Image Studio")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# =========================
# MODELOS
# =========================

MODELO_INICIAL = "google/gemini-2.5-flash-image-preview"

MODELOS_IMAGEM = [
    "google/gemini-2.5-flash-image-preview",
    "openai/gpt-5.4-image-2",
    "black-forest-labs/flux.2-pro",
    "black-forest-labs/flux.2-flex",
    "qwen/qwen3.7-plus",  # deixado aqui, mas pode não gerar imagem de saída
]

# =========================
# PROMPTS PADRÃO
# =========================

PROMPT_COMIC_PADRAO = (
    "Transform the uploaded image into a Western comic book illustration. "
    "Preserve the person's identity, pose, body proportions, composition, and main scene elements. "
    "Convert the image into a detailed comic book style with bold clean ink lines, "
    "rich cel shading, soft halftone texture, expressive contours, strong highlights and shadows, "
    "vibrant but balanced colors, and a polished graphic novel appearance. "
    "Do not make it photorealistic. "
    "The final result should clearly look like a drawn comic book illustration, not a painted photo."
)

NEGATIVE_COMIC_PADRAO = (
    "photorealistic, realistic photo, oil painting, watercolor, blurry, low quality, pixelated, noisy, "
    "anime, manga, cartoon for kids, 3d render, cgi, deformed anatomy, extra limbs, extra fingers, "
    "mutated hands, warped face, asymmetrical eyes, distorted proportions, text, watermark, logo, caption"
)

# =========================
# FUNÇÕES AUXILIARES
# =========================

def pil_to_data_url(img: Image.Image, format_: str = "PNG") -> str:
    """
    Converte uma PIL Image em data URL base64.
    """
    buffer = io.BytesIO()
    img.save(buffer, format=format_)
    mime = "image/png" if format_.upper() == "PNG" else "image/jpeg"
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def data_url_to_pil(data_url: str) -> Image.Image:
    """
    Converte data URL base64 para PIL Image.
    """
    if "," not in data_url:
        raise ValueError("Data URL inválida.")

    _, b64_data = data_url.split(",", 1)
    img_bytes = base64.b64decode(b64_data)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


def extract_images_from_openrouter(data: dict) -> list[Image.Image]:
    """
    Extrai imagens da resposta do OpenRouter.
    """
    imagens = []

    choices = data.get("choices", [])
    if not choices:
        return imagens

    message = choices[0].get("message", {})
    content = message.get("content")

    # content como lista multimodal
    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue

            tipo = item.get("type")

            if tipo == "image_url":
                image_url = item.get("image_url", {})
                url = image_url.get("url", "") if isinstance(image_url, dict) else ""
                if isinstance(url, str) and url.startswith("data:image/"):
                    imagens.append(data_url_to_pil(url))

            elif tipo in ("image", "output_image"):
                url = item.get("url") or item.get("image_url") or ""
                if isinstance(url, dict):
                    url = url.get("url", "")
                if isinstance(url, str) and url.startswith("data:image/"):
                    imagens.append(data_url_to_pil(url))

    # content como string com data URL embutida
    elif isinstance(content, str):
        encontrados = re.findall(
            r"data:image\/[a-zA-Z]+;base64,[A-Za-z0-9+/=]+",
            content
        )
        for data_url in encontrados:
            imagens.append(data_url_to_pil(data_url))

    # fallback adicional
    imagens_msg = message.get("images", [])
    if isinstance(imagens_msg, list):
        for item in imagens_msg:
            if not isinstance(item, dict):
                continue
            image_url = item.get("image_url", {})
            url = image_url.get("url", "") if isinstance(image_url, dict) else ""
            if isinstance(url, str) and url.startswith("data:image/"):
                imagens.append(data_url_to_pil(url))

    return imagens


def download_button_from_pil(img: Image.Image, filename: str, label: str):
    """
    Cria botão de download para imagem PIL.
    """
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    st.download_button(
        label=label,
        data=buf.getvalue(),
        file_name=filename,
        mime="image/png",
    )


def montar_prompt_final(prompt: str, negative_prompt: str, preservar_fundo: bool) -> str:
    """
    Monta o prompt final.
    """
    partes = [prompt.strip()]

    if preservar_fundo:
        partes.append(
            "Preserve the original background and scene layout as much as possible."
        )
    else:
        partes.append(
            "You may simplify the background while preserving the main subject and scene readability."
        )

    if negative_prompt.strip():
        partes.append("Negative prompt / avoid:")
        partes.append(negative_prompt.strip())

    return "\n\n".join(partes)


# =========================
# CHAMADA À API OPENROUTER
# =========================

def gerar_imagem_de_outra_openrouter(
    imagem_pil: Image.Image,
    prompt: str,
    negative_prompt: str,
    model: str,
    size: str = "1024x1024",
    quality: str = "auto",
    preservar_fundo: bool = True,
):
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY não encontrado em st.secrets.")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": APP_REFERER,
        "X-Title": APP_TITLE,
    }

    imagem_data_url = pil_to_data_url(imagem_pil, format_="PNG")
    prompt_final = montar_prompt_final(prompt, negative_prompt, preservar_fundo)

    payload = {
        "model": model,
        "modalities": ["image", "text"],
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_final
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": imagem_data_url
                        }
                    }
                ]
            }
        ],
        "image_config": {
            "size": size,
            "quality": quality
        }
    }

    resp = requests.post(
        OPENROUTER_URL,
        headers=headers,
        json=payload,
        timeout=300,
    )

    if resp.status_code != 200:
        st.error(f"Erro da API OpenRouter — status {resp.status_code}")
        st.code(resp.text)
        raise RuntimeError(f"Falha OpenRouter: {resp.status_code}")

    data = resp.json()
    imagens = extract_images_from_openrouter(data)

    texto_resposta = ""
    try:
        texto_resposta = data["choices"][0]["message"].get("content", "")
    except Exception:
        pass

    return imagens, texto_resposta, data


# =========================
# UI
# =========================

st.subheader("Enviar imagem de referência")

arquivo = st.file_uploader(
    "Escolha uma imagem",
    type=["png", "jpg", "jpeg", "webp"]
)

if arquivo:
    imagem_original = Image.open(arquivo).convert("RGB")
    st.image(imagem_original, caption="Imagem original", use_container_width=True)
else:
    imagem_original = None

st.subheader("Configuração")

modelo = st.selectbox(
    "Modelo OpenRouter",
    MODELOS_IMAGEM,
    index=MODELOS_IMAGEM.index(MODELO_INICIAL),
    help="Escolha um modelo com saída de imagem. O Qwen pode não devolver imagem final."
)

modelo_manual = st.text_input(
    "Ou informe manualmente outro modelo",
    value=""
)

modelo_final = modelo_manual.strip() if modelo_manual.strip() else modelo

col1, col2 = st.columns(2)

with col1:
    prompt_positivo = st.text_area(
        "Prompt positivo",
        value=PROMPT_COMIC_PADRAO,
        height=220,
    )

with col2:
    prompt_negativo = st.text_area(
        "Prompt negativo",
        value=NEGATIVE_COMIC_PADRAO,
        height=220,
    )

col_a, col_b, col_c = st.columns(3)

with col_a:
    tamanho = st.selectbox(
        "Tamanho",
        [
            "1024x1024",
            "768x1024",
            "1024x768",
            "1536x1024",
            "1024x1536",
        ],
        index=0
    )

with col_b:
    qualidade = st.selectbox(
        "Qualidade",
        ["auto", "low", "medium", "high"],
        index=0
    )

with col_c:
    preservar_fundo = st.checkbox(
        "Preservar fundo",
        value=True
    )

st.divider()

with st.expander("Ver prompt final"):
    st.code(montar_prompt_final(prompt_positivo, prompt_negativo, preservar_fundo))

# =========================
# BOTÃO
# =========================

if st.button("🚀 Transformar em comic book"):
    if imagem_original is None:
        st.error("Envie uma imagem primeiro.")
        st.stop()

    if not prompt_positivo.strip():
        st.error("Digite um prompt positivo.")
        st.stop()

    if not OPENROUTER_API_KEY:
        st.error("OPENROUTER_API_KEY não configurada nos secrets.")
        st.stop()

    try:
        st.info("Enviando imagem para o OpenRouter...")

        imagens, texto_resposta, bruto = gerar_imagem_de_outra_openrouter(
            imagem_pil=imagem_original,
            prompt=prompt_positivo,
            negative_prompt=prompt_negativo,
            model=modelo_final,
            size=tamanho,
            quality=qualidade,
            preservar_fundo=preservar_fundo,
        )

        if not imagens:
            st.warning(
                "O modelo respondeu, mas nenhuma imagem foi encontrada na resposta. "
                "Tente outro modelo com saída de imagem."
            )

            if texto_resposta:
                st.markdown("### Resposta textual do modelo")
                st.write(texto_resposta)

            with st.expander("Ver resposta bruta da API"):
                st.json(bruto)

            st.stop()

        st.success("Imagem transformada com sucesso! ✅")

        for idx, img in enumerate(imagens, start=1):
            st.markdown(f"### Resultado {idx}")
            st.image(img, use_container_width=True)

            download_button_from_pil(
                img,
                f"comic_book_result_{idx}.png",
                f"⬇️ Baixar resultado {idx}"
            )

        with st.expander("Ver resposta bruta da API"):
            st.json(bruto)

    except Exception as e:
        st.error(f"Falha ao gerar imagem: {e}")
