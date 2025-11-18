import requests
import streamlit as st

# =========================
# CONFIGURAÇÕES GERAIS
# =========================

API_URL = "https://api.lemonfox.ai/v1/images/generations"

# Senha simples de acesso à página
PASSWORD = "3110"  # 🔒 TROQUE para outra senha antes de publicar


def gerar_imagens(prompt, prompt_negativo, n, tamanho, api_key):
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

    resp = requests.post(API_URL, json=payload, headers=headers)

    # Tratamento de erro com debug
    if resp.status_code != 200:
        st.error(f"Erro da API LemonFox (status {resp.status_code})")
        # Mostra o corpo da resposta (muitas vezes vem mensagem explicando o erro)
        try:
            st.code(resp.text, language="json")
        except Exception:
            st.write(resp.text)
        # Levanta exceção para parar o fluxo de forma controlada
        raise Exception(f"Erro da API LemonFox: {resp.status_code}")

    # Se chegou aqui, status é 200
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
# APP STREAMLIT
# =========================

st.set_page_config(page_title="Gerador SDXL - LemonFox", page_icon="🖼️")

st.title("🖼️ Gerador de Imagens SDXL (LemonFox)")
st.caption("App privado para testes – protegido por senha simples.")

# ---------- BLOQUEIO POR SENHA ----------
senha = st.text_input("Senha de acesso", type="password")

if senha != PASSWORD:
    st.warning("Informe a senha correta para acessar o gerador.")
    st.stop()

st.success("Acesso liberado ✅")

# ---------- API KEY ----------
api_key = st.secrets.get("LEMONFOX_API_KEY", "")

if not api_key:
    st.error(
        "⚠️ API key não encontrada.\n\n"
        "Defina `LEMONFOX_API_KEY` em `.streamlit/secrets.toml` "
        "ou nos secrets do Streamlit Cloud."
    )
    st.stop()

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

    with st.spinner("Gerando imagens na API LemonFox..."):
        try:
            urls = gerar_imagens(
                prompt=prompt_positivo,
                prompt_negativo=prompt_negativo,
                n=n,
                tamanho=tamanho,
                api_key=api_key,
            )
        except Exception as e:
            # A gerar_imagens já mostra detalhes do erro.
            st.error(f"Falha ao gerar imagens: {e}")
            st.stop()

    if not urls:
        st.warning("Nenhuma imagem foi retornada pela API.")
        st.stop()

    st.success(f"{len(urls)} imagem(ns) gerada(s) com sucesso! 🎉")

    # Exibe e disponibiliza download
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
