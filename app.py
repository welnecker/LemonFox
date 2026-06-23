import io
import re
import base64
import requests
from PIL import Image
from huggingface_hub import InferenceClient
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

HF_TOKEN = (
    st.secrets.get("HF_TOKEN", "")
    or st.secrets.get("HUGGINGFACE_API_KEY", "")
)

APP_REFERER = st.secrets.get("APP_REFERER", "https://streamlit.app")
APP_TITLE = st.secrets.get("APP_TITLE", "Comic Book Image Studio")

# =========================
# ENDPOINTS
# =========================

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# =========================
# PROVEDOR
# =========================

provedor = st.selectbox(
    "Provedor",
    ["OpenRouter", "Hugging Face"],
    index=0,
)

# =========================
# MODELOS OPENROUTER
# =========================

MODELO_OPENROUTER_INICIAL = "x-ai/grok-imagine-image-quality"

MODELOS_OPENROUTER_IMAGEM = [
    "x-ai/grok-imagine-image-quality",
    "black-forest-labs/flux.2-max",
    "black-forest-labs/flux.2-klein-4b",
    "bytedance-seed/seedream-4.5",
    "recraft/recraft-v4.1-pro-vector",
    "qwen/qwen3.7-plus",
    "openrouter/auto",
]

# =========================
# MODELOS HUGGING FACE
# =========================

MODELO_HF_INICIAL = "black-forest-labs/FLUX.2-klein-9B"

MODELOS_HF_IMAGEM = [
    "black-forest-labs/FLUX.2-klein-9B",
    "Qwen/Qwen-Image-Edit-2511",
    "autoweeb/Qwen-Image-Edit-2509-Photo-to-Anime",
    "timbrooks/instruct-pix2pix",
    "nitrosocke/comic-diffusion",
    "runwayml/stable-diffusion-v1-5",
    "stabilityai/stable-diffusion-xl-base-1.0",
]

HF_PROVIDER_INICIAL = "replicate"

HF_PROVIDERS = [
    "replicate",
    "fal-ai",
    "wavespeed",
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

def sugerir_provider_hf(model_id: str) -> str:
    """
    Sugere provider Hugging Face de acordo com o modelo.
    """
    if model_id == "Qwen/Qwen-Image-Edit-2511":
        return "fal-ai"

    if model_id == "black-forest-labs/FLUX.2-klein-9B":
        return "replicate"

    return "replicate"


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

    # Caso 2: content como string com data URL
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


# =========================
# CHAMADA OPENROUTER
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
# CHAMADA HUGGING FACE — INFERENCECLIENT
# =========================

def gerar_imagem_huggingface_img2img(
    imagem_pil: Image.Image,
    prompt: str,
    negative_prompt: str,
    model_id: str,
    provider: str = "replicate",
    strength: float = 0.55,
    guidance_scale: float = 7.5,
    preservar_fundo: bool = True,
):
    """
    Chama Hugging Face Inference Providers via huggingface_hub.InferenceClient.
    Tenta input posicional e depois image=...
    """
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN ou HUGGINGFACE_API_KEY não encontrado em st.secrets.")

    prompt_final = montar_prompt_final(
        prompt=prompt,
        negative_prompt=negative_prompt,
        preservar_fundo=preservar_fundo,
    )

    buffer = io.BytesIO()
    imagem_pil.save(buffer, format="PNG")
    input_image = buffer.getvalue()

    client = InferenceClient(
        provider=provider,
        api_key=HF_TOKEN,
    )

    erros = []

    # Tentativa 1 — igual aos exemplos oficiais
    try:
        image = client.image_to_image(
            input_image,
            prompt=prompt_final,
            model=model_id,
        )

        if isinstance(image, Image.Image):
            return [image.convert("RGB")], {
                "provider": "huggingface",
                "hf_provider": provider,
                "model": model_id,
                "method": "positional_input_image",
                "strength": strength,
                "guidance_scale": guidance_scale,
            }

        erros.append(f"Tentativa posicional retornou tipo inesperado: {type(image)}")

    except Exception as e:
        erros.append(f"Tentativa posicional falhou: {repr(e)}")

    # Tentativa 2 — keyword image=...
    try:
        image = client.image_to_image(
            image=input_image,
            prompt=prompt_final,
            model=model_id,
        )

        if isinstance(image, Image.Image):
            return [image.convert("RGB")], {
                "provider": "huggingface",
                "hf_provider": provider,
                "model": model_id,
                "method": "keyword_image",
                "strength": strength,
                "guidance_scale": guidance_scale,
            }

        erros.append(f"Tentativa image= retornou tipo inesperado: {type(image)}")

    except Exception as e:
        erros.append(f"Tentativa image= falhou: {repr(e)}")

    raise RuntimeError(
        "Falha ao chamar Hugging Face InferenceClient.\n\n"
        f"Provider: {provider}\n"
        f"Modelo: {model_id}\n\n"
        "Erros:\n- " + "\n- ".join(erros)
    )


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

    hf_provider = None

else:
    modelo = st.selectbox(
        "Modelo Hugging Face",
        MODELOS_HF_IMAGEM,
        index=MODELOS_HF_IMAGEM.index(MODELO_HF_INICIAL),
        help="Modelo chamado via Hugging Face Inference Providers."
    )

    modelo_manual = st.text_input(
        "Ou informe manualmente outro modelo Hugging Face",
        value="",
        placeholder="ex: black-forest-labs/FLUX.2-klein-9B"
    )

    modelo_final_preview = modelo_manual.strip() if modelo_manual.strip() else modelo
    provider_recomendado = sugerir_provider_hf(modelo_final_preview)

    hf_provider = st.selectbox(
        "Provider Hugging Face",
        HF_PROVIDERS,
        index=HF_PROVIDERS.index(provider_recomendado)
        if provider_recomendado in HF_PROVIDERS
        else HF_PROVIDERS.index(HF_PROVIDER_INICIAL),
        help="Provider usado pelo InferenceClient da Hugging Face."
    )

modelo_final = modelo_manual.strip() if modelo_manual.strip() else modelo

st.caption(f"Provedor selecionado: `{provedor}`")
st.caption(f"Modelo selecionado: `{modelo_final}`")

if provedor == "Hugging Face":
    provider_recomendado = sugerir_provider_hf(modelo_final)

    st.caption(f"HF provider selecionado: `{hf_provider}`")
    st.caption(f"HF provider recomendado para este modelo: `{provider_recomendado}`")

    if modelo_final == "Qwen/Qwen-Image-Edit-2511":
        st.warning(
            "Qwen/Qwen-Image-Edit-2511 está disponível como opção experimental. "
            "Ele usa provider recomendado fal-ai, mas pode falhar no InferenceClient. "
            "Para estabilidade, prefira black-forest-labs/FLUX.2-klein-9B com replicate."
        )

    if modelo_final == "autoweeb/Qwen-Image-Edit-2509-Photo-to-Anime":
        st.warning(
            "Este modelo é um LoRA Photo-to-Anime finetuned do Qwen-Image-Edit-2509. "
            "O provider recomendado é wavespeed. Se der Bad Request, o problema provavelmente "
            "é o provider/roteamento, não o app. Use prompt simples, como: transform into anime."
        )

    if hf_provider != provider_recomendado:
        st.warning(
            f"O provider selecionado foi `{hf_provider}`, mas para `{modelo_final}` "
            f"o recomendado é `{provider_recomendado}`. "
            f"Na geração, o app usará `{provider_recomendado}` automaticamente."
        )

# =========================
# UI — PROMPTS
# =========================

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
                "Alto transforma mais, mas pode perder identidade. "
                "Pode ser ignorado por alguns providers."
            )
        )

    with col_hf2:
        guidance_scale = st.slider(
            "Guidance scale",
            min_value=1.0,
            max_value=15.0,
            value=7.5,
            step=0.5,
            help=(
                "Quanto o modelo obedece ao prompt. "
                "Pode ser ignorado por alguns providers."
            )
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

    if provedor == "Hugging Face" and not HF_TOKEN:
        st.error("HF_TOKEN ou HUGGINGFACE_API_KEY não configurado nos secrets.")
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
            provider_usado = sugerir_provider_hf(modelo_final)

            st.info(f"HF provider usado: {provider_usado}")

            imagens, bruto = gerar_imagem_huggingface_img2img(
                imagem_pil=imagem_original,
                prompt=prompt_positivo,
                negative_prompt=prompt_negativo,
                model_id=modelo_final,
                provider=provider_usado,
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
