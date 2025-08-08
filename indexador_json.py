import os
import json
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

def iter_convenio_registros(dados):
    """
    Normaliza diferentes formatos de JSON de convênios e
    produz tuplas (convenio_nome, info_dict_com_modalidades).
    Aceita:
      1) dict: {"Unimed": {"modalidades": {...}}, "Bradesco": {...}}
      2) list de dicts com campos nomeados:
         [{"convenio": "Unimed", "modalidades": {...}}, ...]
      3) list de dicts single-key:
         [{"Unimed": {"modalidades": {...}}}, {"Bradesco": {...}}]
    """
    # Caso 1: dict no topo
    if isinstance(dados, dict):
        for convenio, info in dados.items():
            if isinstance(info, dict):
                yield str(convenio), info
        return

    # Caso 2/3: lista no topo
    if isinstance(dados, list):
        for item in dados:
            # 2) {"convenio": "Unimed", "modalidades": {...}}
            if isinstance(item, dict) and "convenio" in item:
                nome = str(item.get("convenio", "")).strip()
                info = item
                if nome and isinstance(info, dict):
                    yield nome, info
                continue

            # 3) {"Unimed": {"modalidades": {...}}}
            if isinstance(item, dict) and len(item) == 1:
                nome = str(next(iter(item.keys())))
                info = next(iter(item.values()))
                if isinstance(info, dict):
                    yield nome, info
                continue
    # Qualquer outro formato é ignorado silenciosamente
# Carrega chave da OpenAI do .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Caminhos
PASTA_JSON = "base_conhecimento"
PASTA_VETOR = "vetor/faiss_index"

# Inicializa embedding
embeddings = OpenAIEmbeddings()

# Lista para armazenar todos os documentos
documentos = []

convenio_docs_count = 0  # <--- adicione no topo do arquivo, antes do loop

# Itera sobre arquivos .json da base
for caminho in Path(PASTA_JSON).glob("*.json"):
    with open(caminho, "r", encoding="utf-8") as f:
        dados = json.load(f)

        # ================= INÍCIO BLOCO ESPECIAL: CONVÊNIOS =================
        nome_arquivo = caminho.name.lower()

        # Detecta arquivos de convênio pelo nome e/ou pelo conteúdo
        eh_convenio = (
            ("conven" in nome_arquivo) or
            ("convênio" in nome_arquivo) or
            ("regras" in nome_arquivo and ("conv" in nome_arquivo or "plano" in nome_arquivo)) or
            nome_arquivo == "regras_de_convenio.json"
        )

        if not eh_convenio and isinstance(dados, (dict, list)):
            def contem_modalidades(x: dict) -> bool:
                return (
                    isinstance(x, dict) and (
                        "modalidades" in x or
                        "modalidade" in x or
                        ("convenio" in x and ("modalidades" in x or "modalidade" in x))
                    )
                )
            if isinstance(dados, dict):
                eh_convenio = any(contem_modalidades(v) for v in dados.values())
                if not eh_convenio:
                    # wrappers comuns
                    for k in ["convenios", "dados", "regras", "regras_convenios", "data"]:
                        if k in dados and isinstance(dados[k], (dict, list)):
                            alvo = dados[k]
                            eh_convenio = (
                                (isinstance(alvo, dict) and any(contem_modalidades(v) for v in alvo.values())) or
                                (isinstance(alvo, list) and any(contem_modalidades(i) for i in alvo if isinstance(i, dict)))
                            )
                            if eh_convenio:
                                break
            else:  # list
                eh_convenio = any(contem_modalidades(x) for x in dados if isinstance(x, dict))

        if eh_convenio:
            print(f"🔎 Detectado arquivo de convênios: {caminho.name}")

            def add_doc(convenio_nome: str, modalidade_nome: str, dados_modalidade: dict):
                global convenio_docs_count
                convenio_norm = str(convenio_nome).strip().lower()
                modalidade_norm = str(modalidade_nome).strip().lower()

                cobre = dados_modalidade.get("cobre")
                excecoes = dados_modalidade.get("excecoes", [])
                observacoes = dados_modalidade.get("observacoes", [])
                sinonimos = dados_modalidade.get("sinonimos", [])
                if isinstance(sinonimos, str):
                    sinonimos = [sinonimos]

                page = (
                    "TIPO: COBERTURA DE CONVÊNIO\n"
                    f"Convênio: {convenio_nome}\n"
                    f"Modalidade: {modalidade_nome}\n"
                    f"Cobertura: {'SIM' if cobre else 'NÃO' if cobre is not None else 'indefinido'}\n"
                    f"Exceções: {', '.join(map(str, excecoes)) if excecoes else '—'}\n"
                    f"Observações: {', '.join(map(str, observacoes)) if observacoes else '—'}\n"
                    f"Sinônimos_modalidade: {', '.join(s.strip().lower() for s in sinonimos if isinstance(s, str))}\n"
                    "Palavras-chave: cobertura de convênio, cobre, autorizado, autorização, plano de saúde, aceita\n"
                )

                documentos.append(
                    Document(
                        page_content=page,
                        metadata={
                            "tipo": "convenio",
                            "convenio": convenio_norm,
                            "modalidade": modalidade_norm,
                            "origem": caminho.name,
                        },
                    )
                )
                convenio_docs_count += 1

            # 1) FORMATO FLAT (seu arquivo: lista de registros)
            if isinstance(dados, list) and all(isinstance(x, dict) for x in dados):
                for reg in dados:
                    if "convenio" in reg and "modalidade" in reg:
                        add_doc(reg["convenio"], reg["modalidade"], reg)
                continue  # não processa este arquivo no bloco padrão

            # 2) FORMATO DICIONÁRIO por convênio
            if isinstance(dados, dict):
                # wrappers como {"convenios":[...]}, {"regras": {...}}, etc.
                for wrap_key in ["convenios", "dados", "regras", "regras_convenios", "data"]:
                    if wrap_key in dados and isinstance(dados[wrap_key], (dict, list)):
                        dados = dados[wrap_key]
                        break

                if isinstance(dados, dict):
                    for convenio, info in dados.items():
                        if not isinstance(info, dict):
                            continue
                        modalidades = info.get("modalidades") or info.get("modalidade") or {}
                        if isinstance(modalidades, str):  # normaliza string para dict unitário
                            modalidades = {modalidades: info}
                        if not isinstance(modalidades, dict):
                            continue
                        for modalidade, dados_modalidade in modalidades.items():
                            if isinstance(dados_modalidade, dict):
                                add_doc(convenio, modalidade, dados_modalidade)
                    continue

                if isinstance(dados, list):
                    for reg in dados:
                        if isinstance(reg, dict) and "convenio" in reg and "modalidade" in reg:
                            add_doc(reg["convenio"], reg["modalidade"], reg)
                    continue
        # ================= FIM BLOCO ESPECIAL: CONVÊNIOS =================


    # Garante que os dados sejam uma lista
    if isinstance(dados, list):
        for item in dados:
            # Usa o campo "exame" como título e concatena com todos os textos de preparo
            nome_exame = item.get("exame", "").strip()
            # Garante que sempre teremos uma lista
            valor_bruto = item.get("preparo", [])

            if valor_bruto is None:
                textos_preparo = []
            elif isinstance(valor_bruto, str):
                textos_preparo = [valor_bruto]
            elif isinstance(valor_bruto, list):
                textos_preparo = valor_bruto
            else:
                textos_preparo = []

            sinonimos = item.get("sinonimos", [])
            if isinstance(sinonimos, str):
                sinonimos = [sinonimos]
            elif not isinstance(sinonimos, list):
                sinonimos = []

            # Novo bloco: observações adicionais de crianças
            observacoes_adicionais = item.get("observacoes_adicionais", {})
            orientacoes_crianca = observacoes_adicionais.get("orientacoes_crianca", [])
            if isinstance(orientacoes_crianca, str):
                orientacoes_crianca = [orientacoes_crianca]
            elif not isinstance(orientacoes_crianca, list):
                orientacoes_crianca = []


            codigo = item.get("codigo", "")
            medicos = item.get("medicos_envolvidos", []) or item.get("medicos", [])  # compatibilidade
            medicos_str = f"Médicos envolvidos: {', '.join(medicos)}" if medicos else ""

            # Se não houver código e médicos, mas for um item válido (como orientações gerais)
            if not codigo and not medicos and "preparo" in item:
                medicos_str = "Orientações gerais"


            # Concatena tudo
            blocos = [nome_exame, f"Código: {codigo}"]
            if medicos_str:
                blocos.append(medicos_str)

            texto_unificado = "\n".join(blocos + textos_preparo + orientacoes_crianca + sinonimos).strip()


            if texto_unificado:
                doc = Document(
                    page_content=texto_unificado,
                    metadata={"origem": caminho.name}
                )
                documentos.append(doc)

print(f"✅ Total de blocos indexados: {len(documentos)}")
print(f"🏷️  Blocos de convênios indexados: {convenio_docs_count}")

# Cria e salva índice vetorizado
db = FAISS.from_documents(documentos, embeddings)
db.save_local(PASTA_VETOR)
print(f"💾 Vetor salvo em: {PASTA_VETOR}")
