import requests
import streamlit as st

# =========================
# CONFIGURAÇÕES GERAIS
# =========================

# API LemonFox
LEMONFOX_API_URL = "https://api.lemonfox.ai/v1/images/generations"

# API Hugging Face Inference (NOVO endpoint)
HF_API_URL = "https://router.huggingface.co/hf-inference"
HF_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"  # você pode trocar por outro modelo depois

# Senha simples de acesso à página
PASSWORD = "1234"  # 🔒 TROQUE para outra senha antes de publicar


# =========================
# FUNÇÕES – LEMONFOX
# =========================

def gerar_imagens_lemonfox(prompt, prompt_negativo, n, tamanho, api_key):
    """
    Chama a API do LemonFox para gerar imagens.
    Retorna uma lista de URLs.
    Em caso de erro, mostra o corpo da resposta para debug.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "prompt": prompt,
        "n": n,
        "tamanho": tamanho,
        "formato_de_resposta": "url",
    }

    if prompt_negativo:
        payload["prompt_negativo"] = prompt_negativo

    resp = requests.post(LEMONFOX_API_URL, json=payload, headers=headers)

    # Tratamento de erro com debug
    if resp.status_code != 200:
        st.error(f"Erro da API LemonFox (status {resp.status_code})")
        try:
            st.code(resp.text, language="json")
        except Exception:
            st.write(resp.text)
        raise Exception(f"Erro da API LemonFox: {resp.status_code}")

    data = resp.json()
    urls = [item["url"] for item in data.get("data", [])]
    return urls


def baixar_bytes_imagem(url: str) -> bytes:
    """
    Faz o download da imagem a partir da URL.
    Retorna os bytes da imagem.
    """
    r = requests.get(url)
    r.raise_for_status()
    return r.content


# =========================
# FUNÇÕES – HUGGING FACE (router.huggingface.co)
# =========================

def gerar_imagens_hf(prompt, prompt_negativo, n, tamanho, hf_token):
    """
    Chama a API de Inference da Hugging Face (router.huggingface.co)
    para gerar imagens com o modelo HF_MODEL_ID.
    Retorna uma lista de bytes (cada item é o conteúdo da imagem).
    """
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Accept": "image/png",
        "Content-Type": "application/json",
    }

    # Converte "1024x1024" -> width=1024, height=1024
    try:
        w_str, h_str = tamanho.lower().split("x")
        width = int(w_str)
        height = int(h_str)
    except Exception:
        width, height = 1024, 1024

    url = HF_API_URL
    imagens_bytes = []

    # Para evitar dor de cabeça, geramos 1 por vez no loop
    for i in range(n):
        payload = {
            "model": HF_MODEL_ID,
            "inputs": prompt,
            "parameters": {
                "negative_prompt": prompt_negativo or "",
                "width": width,
                "height": height,
                "guidance_scale": 7.0,
                "num_inference_steps": 30,
            }
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=120)

        if resp.status_code != 200:
            st.error(f"Erro da API Hugging Face (status {resp.status_code}) na imagem {i+1}")
            try:
                st.code(resp.text, language="json")
            except Exception:
                st.write(resp.text)
            raise Exception(f"Erro da API Hugging Face: {resp.status_code}")

        imagens_bytes.append(resp.content)

    return imagens_bytes


# =========================
# APP STREAMLIT
# =========================

st.set_page_config(
    page_title="Gerador de Imagens – LemonFox / Hugging Face",
    page_icon="🖼️"
)

st.title("🖼️ Gerador de Imagens")
st.caption("App privado para testes – protegido por senha simples.")

# ---------- BLOQUEIO POR SENHA ----------
senha = st.text_input("Senha de acesso", type="password")

if senha != PASSWORD:
    st.warning("Informe a senha correta para acessar o gerador.")
    st.stop()

st.success("Acesso liberado ✅")

# ---------- SECRETS ----------
lemonfox_api_key = st.secrets.get("LEMONFOX_API_KEY", "")
hf_api_token = st.secrets.get("HF_API_TOKEN", "")

# ---------- ESCOLHA DO PROVIDER ----------
st.subheader("Provedor de geração")

provider = st.radio(
    "Escolha qual API usar:",
    [
        "LemonFox (SDXL API)",
        "Hugging Face (SDXL / Stable Diffusion)",
    ],
)

if provider.startswith("LemonFox"):
    if not lemonfox_api_key:
        st.error(
            "⚠️ LEMONFOX_API_KEY não definida nos secrets.\n\n"
            "Adicione no secrets.toml ou nos Secrets do Streamlit Cloud."
        )
elif provider.startswith("Hugging Face"):
    if not hf_api_token:
        st.error(
            "⚠️ HF_API_TOKEN não definido nos secrets.\n\n"
            "Crie um token na sua conta da Hugging Face e adicione nos secrets."
        )

# ---------- INTERFACE DE PROMPTS ----------
st.subheader("Prompt de geração")

prompt_positivo = st.text_area(
    "Prompt positivo (o que você quer na imagem):",
    height=140,
    placeholder=(
        "ex: Laura Massariol, stunning Brazilian redhead woman, late 20s, "
        "long wavy fiery copper hair, green eyes, comic book style, "
        "full body, curves, dramatic warm lighting, ultra detailed"
    ),
)

prompt_negativo = st.text_area(
    "Prompt negativo (o que você NÃO quer):",
    height=100,
    placeholder=(
        "ex: low quality, blurry, pixelated, bad anatomy, extra limbs, "
        "flat butt, flat chest, deformed face, extra fingers, text, watermark"
    ),
)

col1, col2 = st.columns(2)
with col1:
    n = st.slider("Quantidade de imagens", min_value=1, max_value=4, value=1)
with col2:
    tamanho = st.selectbox(
        "Tamanho da imagem",
        ["512x512", "768x768", "1024x1024"],
        index=2,
    )

gerar = st.button("🚀 Gerar imagens")

if gerar:
    if not prompt_positivo.strip():
        st.warning("Digite pelo menos o prompt positivo para gerar as imagens.")
        st.stop()

    # Verifica se o provider escolhido tem credencial
    if provider.startswith("LemonFox") and not lemonfox_api_key:
        st.error("LEMONFOX_API_KEY não configurada. Não é possível usar LemonFox.")
        st.stop()
    if provider.startswith("Hugging Face") and not hf_api_token:
        st.error("HF_API_TOKEN não configurado. Não é possível usar Hugging Face.")
        st.stop()

    with st.spinner(f"Gerando imagens com {provider}..."):
        try:
            if provider.startswith("LemonFox"):
                # Retorna URLs
                urls = gerar_imagens_lemonfox(
                    prompt=prompt_positivo,
                    prompt_negativo=prompt_negativo,
                    n=n,
                    tamanho=tamanho,
                    api_key=lemonfox_api_key,
                )

                if not urls:
                    st.warning("Nenhuma imagem foi retornada pela API LemonFox.")
                    st.stop()

                st.success(f"{len(urls)} imagem(ns) gerada(s) com sucesso pela LemonFox! 🎉")

                for i, url in enumerate(urls, start=1):
                    st.markdown(f"### Imagem {i}")
                    st.image(url, caption=f"Imagem {i}", use_column_width=True)

                    try:
                        img_bytes = baixar_bytes_imagem(url)
                        st.download_button(
                            label=f"⬇️ Baixar imagem {i}",
                            data=img_bytes,
                            file_name=f"lemonfox_img_{i}.png",
                            mime="image/png",
                        )
                    except Exception as e:
                        st.error(f"Não foi possível preparar o download da imagem {i}: {e}")

            else:
                # Hugging Face – retorna bytes
                imagens_bytes = gerar_imagens_hf(
                    prompt_positivo,
                    prompt_negativo,
                    n,
                    tamanho,
                    hf_api_token,
                )

                if not imagens_bytes:
                    st.warning("Nenhuma imagem foi retornada pela API Hugging Face.")
                    st.stop()

                st.success(f"{len(imagens_bytes)} imagem(ns) gerada(s) com sucesso pela Hugging Face! 🎉")

                for i, img_bytes in enumerate(imagens_bytes, start=1):
                    st.markdown(f"### Imagem {i}")
                    # st.image aceita bytes diretamente
                    st.image(img_bytes, caption=f"Imagem {i}", use_column_width=True)

                    st.download_button(
                        label=f"⬇️ Baixar imagem {i}",
                        data=img_bytes,
                        file_name=f"huggingface_img_{i}.png",
                        mime="image/png",
                    )

        except Exception as e:
            st.error(f"Falha ao gerar imagens: {e}")
            st.stop()
