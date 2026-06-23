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
st.caption("Transforme uma imagem em estilo comic book usando OpenRouter ou Hugging Face")

# ---- Senha simples de acesso ----
APP_PASSWORD = st.secrets.get("APP_PASSWORD", "1234")

senha = st.text_input("Senha de acesso", type="password")

if senha != APP_PASSWORD:
    st.warning("Digite a senha correta para acessar o gerador de imagens.")
    st.stop()

st.success("Acesso liberado! ✅")

# =========================
# SECRETS / CHAVES
# =========================

OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "")
HUGGINGFACE_API_KEY = st.secrets.get("HUGGINGFACE_API_KEY", "")

APP_REFERER = st.secrets.get("APP_REFERER", "https://streamlit.app")
APP_TITLE = st.secrets.get("APP_TITLE", "Comic Book Image Studio")

# =========================
# ENDPOINTS
# =========================

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

HF_IMAGE_TO_IMAGE_URL_TEMPLATE = (
    "https://api-inference.huggingface.co/models/{model_id}"
)

# =========================
# PROVEDOR
# =========================

provedor = st.selectbox(
    "Provedor",
    ["OpenRouter", "Hugging Face"],
    index=0,
)

# =========================
# MODELOS
# =========================

MODELO_OPENROUTER_INICIAL = "x-ai/grok-imagine-image-quality"

MODELOS_OPENROUTER_IMAGEM = [
    "x-ai/grok-imagine-image-quality",
    "black-forest-labs/flux.2-max",
    "x-ai/grok-imagine-image-quality:free",
    "recraft/recraft-v4.1-pro-vector",
    "qwen/qwen3.7-plus",
]

MODELO_HF_INICIAL = "timbrooks/instruct-pix2pix"

MODELOS_HF_IMAGEM = [
    "timbrooks/instruct-pix2pix",
    "runwayml/stable-diffusion-v1-5",
    "stabilityai/stable-diffusion-xl-base-1.0",
    "nitrosocke/comic-diffusion",
    "lllyasviel/sd-controlnet-canny",
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
    "photorealistic, realistic photo, photo filter, oil painting, watercolor, blurry, low quality, pixelated, noisy, "
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


def base64_puro_para_imagem(b64_data: str) -> Image.Image:
    """
    Converte base64 puro para PIL Image.
    """
    img_bytes = base64.b64decode(b64_data)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


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
    Monta o prompt final enviado ao modelo.
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


def extract_images_from_openrouter(data: dict) -> list[Image.Image]:
    """
    Extrai imagens da resposta do OpenRouter.
    Cobre formatos comuns:
    - message.content como lista multimodal
    - message.content como texto com data URL
    - message.images
    """
    imagens = []

    choices = data.get("choices", [])
    if not choices:
        return imagens

    message = choices[0].get("message", {})
    content = message.get("content")

    # Caso 1: content como lista multimodal
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

    # Caso 2: content como texto contendo data URL
    elif isinstance(content, str):
        encontrados = re.findall(
            r"data:image\/[a-zA-Z]+;base64,[A-Za-z0-9+/=]+",
            content
        )

        for data_url in encontrados:
            imagens.append(data_url_to_pil(data_url))

    # Caso 3: fallback message.images
    imagens_msg = message.get("images", [])

    if isinstance(imagens_msg, list):
        for item in imagens_msg:
            if not isinstance(item, dict):
                continue

            image_url = item.get("image_url", {})
            url = ""

            if isinstance(image_url, dict):
                url = image_url.get("url", "")
            elif isinstance(image_url, str):
                url = image_url

            if isinstance(url, str) and url.startswith("data:image/"):
                imagens.append(data_url_to_pil(url))

    return imagens


def extract_images_from_huggingface_json(bruto) -> list[Image.Image]:
    """
    Tenta extrair imagem de respostas JSON da Hugging Face.
    Alguns endpoints retornam bytes de imagem; outros podem retornar base64.
    """
    imagens = []

    if isinstance(bruto, dict):
        possiveis_chaves = [
            "b64_json",
            "image",
            "generated_image",
            "output",
        ]

        for chave in possiveis_chaves:
            valor = bruto.get(chave)

            if isinstance(valor, str):
                if valor.startswith("data:image/"):
                    imagens.append(data_url_to_pil(valor))
                else:
                    try:
                        imagens.append(base64_puro_para_imagem(valor))
                    except Exception:
                        pass

            elif isinstance(valor, list):
                for item in valor:
                    if isinstance(item, str):
                        if item.startswith("data:image/"):
                            imagens.append(data_url_to_pil(item))
                        else:
                            try:
                                imagens.append(base64_puro_para_imagem(item))
                            except Exception:
                                pass

    elif isinstance(bruto, list):
        for item in bruto:
            if isinstance(item, dict):
                imagens.extend(extract_images_from_huggingface_json(item))
            elif isinstance(item, str):
                if item.startswith("data:image/"):
                    imagens.append(data_url_to_pil(item))
                else:
                    try:
                        imagens.append(base64_puro_para_imagem(item))
                    except Exception:
                        pass

    return imagens


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
        "modalities": ["image"],
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

    if resp.status_code == 404:
        st.error(
            "Erro 404: o modelo existe, mas não aceitou a modalidade solicitada. "
            "Para modelos de imagem pura, este script usa modalities=['image']."
        )
        st.code(resp.text)
        raise RuntimeError(f"Modalidade incompatível no OpenRouter: {model}")

    if resp.status_code != 200:
        st.error(f"Erro da API OpenRouter — status {resp.status_code}")
        st.code(resp.text)
        raise RuntimeError(f"Falha OpenRouter: {resp.status_code}")

    data = resp.json()
    imagens = extract_images_from_openrouter(data)

    return imagens, data


# =========================
# CHAMADA À API HUGGING FACE
# =========================

def gerar_imagem_huggingface_img2img(
    imagem_pil: Image.Image,
    prompt: str,
    negative_prompt: str,
    model_id: str,
    strength: float = 0.55,
    guidance_scale: float = 7.5,
    preservar_fundo: bool = True,
):
    if not HUGGINGFACE_API_KEY:
        raise RuntimeError("HUGGINGFACE_API_KEY não encontrado em st.secrets.")

    url = HF_IMAGE_TO_IMAGE_URL_TEMPLATE.format(model_id=model_id)

    headers = {
        "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
    }

    prompt_final = montar_prompt_final(
        prompt=prompt,
        negative_prompt=negative_prompt,
        preservar_fundo=preservar_fundo,
    )

    buffer = io.BytesIO()
    imagem_pil.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()

    files = {
        "image": ("input.png", image_bytes, "image/png"),
    }

    data = {
        "prompt": prompt_final,
        "negative_prompt": negative_prompt,
        "strength": str(strength),
        "guidance_scale": str(guidance_scale),
    }

    resp = requests.post(
        url,
        headers=headers,
        files=files,
        data=data,
        timeout=300,
    )

    content_type = resp.headers.get("content-type", "")

    if resp.status_code != 200:
        st.error(f"Erro Hugging Face — status {resp.status_code}")

        if "application/json" in content_type:
            try:
                st.json(resp.json())
            except Exception:
                st.code(resp.text)
        else:
            st.code(resp.text)

        raise RuntimeError(f"Falha Hugging Face: {resp.status_code}")

    # Muitos endpoints retornam imagem direta
    if content_type.startswith("image/"):
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        return [img], {
            "provider": "huggingface",
            "model": model_id,
            "content_type": content_type,
        }

    # Alguns retornam JSON
    if "application/json" in content_type:
        bruto = resp.json()
        imagens = extract_images_from_huggingface_json(bruto)
        return imagens, bruto

    raise RuntimeError(f"Resposta Hugging Face em formato não reconhecido: {content_type}")


# =========================
# UI — IMAGEM
# =========================

st.subheader("Enviar imagem de referência")

arquivo = st.file_uploader(
    "Escolha uma imagem",
    type=["png", "jpg", "jpeg", "webp"]
)

if arquivo:
    imagem_original = Image.open(arquivo).convert("RGB")
    st.image(
        imagem_original,
        caption="Imagem original",
        use_column_width=True
    )
else:
    imagem_original = None

# =========================
# UI — CONFIGURAÇÃO
# =========================

st.subheader("Configuração")

if provedor == "OpenRouter":
    modelo = st.selectbox(
        "Modelo OpenRouter",
        MODELOS_OPENROUTER_IMAGEM,
        index=MODELOS_OPENROUTER_IMAGEM.index(MODELO_OPENROUTER_INICIAL),
        help=(
            "Modelos chamados pelo OpenRouter. "
            "Para image-to-image, use modelos que aceitam imagem de referência e saída de imagem."
        )
    )

    modelo_manual = st.text_input(
        "Ou informe manualmente outro modelo OpenRouter",
        value="",
        placeholder="ex: x-ai/grok-imagine-image-quality"
    )

else:
    modelo = st.selectbox(
        "Modelo Hugging Face",
        MODELOS_HF_IMAGEM,
        index=MODELOS_HF_IMAGEM.index(MODELO_HF_INICIAL),
        help=(
            "Modelos chamados pela Hugging Face. "
            "Nem todo modelo do Hub aceita image-to-image via endpoint direto."
        )
    )

    modelo_manual = st.text_input(
        "Ou informe manualmente outro modelo Hugging Face",
        value="",
        placeholder="ex: timbrooks/instruct-pix2pix"
    )

modelo_final = modelo_manual.strip() if modelo_manual.strip() else modelo

st.caption(f"Provedor selecionado: `{provedor}`")
st.caption(f"Modelo selecionado: `{modelo_final}`")

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

# =========================
# UI — PARÂMETROS COMUNS
# =========================

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
        index=0,
        help="Usado principalmente pelo OpenRouter. Alguns modelos podem ignorar."
    )

with col_b:
    qualidade = st.selectbox(
        "Qualidade",
        ["auto", "low", "medium", "high"],
        index=0,
        help="Usado principalmente pelo OpenRouter. Alguns modelos podem ignorar."
    )

with col_c:
    preservar_fundo = st.checkbox(
        "Preservar fundo",
        value=True
    )

# =========================
# UI — PARÂMETROS HUGGING FACE
# =========================

if provedor == "Hugging Face":
    st.subheader("Parâmetros Hugging Face")

    col_hf1, col_hf2 = st.columns(2)

    with col_hf1:
        strength = st.slider(
            "Strength / intensidade da transformação",
            min_value=0.10,
            max_value=0.95,
            value=0.55,
            step=0.05,
            help=(
                "Baixo preserva mais a imagem original. "
                "Alto transforma mais, mas pode perder identidade."
            )
        )

    with col_hf2:
        guidance_scale = st.slider(
            "Guidance scale",
            min_value=1.0,
            max_value=15.0,
            value=7.5,
            step=0.5,
            help="Quanto o modelo obedece ao prompt."
        )

else:
    strength = None
    guidance_scale = None

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

    if provedor == "OpenRouter" and not OPENROUTER_API_KEY:
        st.error("OPENROUTER_API_KEY não configurada nos secrets.")
        st.stop()

    if provedor == "Hugging Face" and not HUGGINGFACE_API_KEY:
        st.error("HUGGINGFACE_API_KEY não configurada nos secrets.")
        st.stop()

    try:
        st.info(f"Enviando imagem para {provedor}...")
        st.info(f"Modelo usado: {modelo_final}")

        if provedor == "OpenRouter":
            imagens, bruto = gerar_imagem_de_outra_openrouter(
                imagem_pil=imagem_original,
                prompt=prompt_positivo,
                negative_prompt=prompt_negativo,
                model=modelo_final,
                size=tamanho,
                quality=qualidade,
                preservar_fundo=preservar_fundo,
            )

        else:
            imagens, bruto = gerar_imagem_huggingface_img2img(
                imagem_pil=imagem_original,
                prompt=prompt_positivo,
                negative_prompt=prompt_negativo,
                model_id=modelo_final,
                strength=strength,
                guidance_scale=guidance_scale,
                preservar_fundo=preservar_fundo,
            )

        if not imagens:
            st.warning(
                "O modelo respondeu, mas nenhuma imagem foi encontrada na resposta. "
                "Veja a resposta bruta abaixo para identificar o formato retornado pelo provedor."
            )

            with st.expander("Ver resposta bruta da API"):
                st.json(bruto)

            st.stop()

        st.success("Imagem transformada com sucesso! ✅")

        for idx, img in enumerate(imagens, start=1):
            st.markdown(f"### Resultado {idx}")
            st.image(img, use_column_width=True)

            download_button_from_pil(
                img,
                f"comic_book_result_{idx}.png",
                f"⬇️ Baixar resultado {idx}"
            )

        with st.expander("Ver resposta bruta da API"):
            st.json(bruto)

    except Exception as e:
        st.error(f"Falha ao gerar imagem: {e}")
