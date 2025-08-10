import os
import json
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# ========================= CONFIG =========================
PASTA_JSON = "base_conhecimento"      # pasta dos .json
PASTA_VETOR = "vetor/faiss_index"     # saída FAISS
INDICE_CONVENIOS_JSON = "vetor/indice_convenios.json"
# ==========================================================

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --------- normalização de modalidades / sinônimos ---------
MODALIDADE_MAP = {
    "ressonancia": "RM", "ressonância": "RM", "rm": "RM",
    "ressonancia magnetica": "RM", "ressonância magnética": "RM",
    "angio rm": "ANGIO-RM", "angio-rm": "ANGIO-RM",

    "tomografia": "TC", "tc": "TC",
    "angio tc": "ANGIO-TC", "angio-tc": "ANGIO-TC",

    "raio x": "RX", "raio-x": "RX", "rx": "RX",

    "ultrassom": "US", "us": "US", "usg": "US", "ecografia": "US",

    "eeg": "EEG", "eletroencefalograma": "EEG",
    "enmg": "ENMG", "eletroneuromiografia": "ENMG",

    "polissonografia": "POLISSONOGRAFIA", "psg": "POLISSONOGRAFIA",

    "rm de mama": "RM DE MAMA",
    "rm de mastoide": "RM DE MASTOIDE",
    "rm de face": "RM DE FACE",
    "rm de orbita": "RM DE ÓRBITA", "rm de órbita": "RM DE ÓRBITA",

    "tc (exceto angio-tc)": "TC (EXCETO ANGIO-TC)",
}
def norm_modalidade(s: str) -> str:
    s = (s or "").strip()
    if not s: return s
    low = s.lower()
    return MODALIDADE_MAP.get(low, s.upper())

# --------- normalizadores auxiliares ---------
def _to_bool_cobre(v):
    if isinstance(v, bool): return v
    if isinstance(v, (int, float)): return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"sim","true","verdadeiro","yes","y"}: return True
        if s in {"nao","não","false","falso","no","n"}: return False
    return None

def _as_list(x):
    if x is None: return []
    if isinstance(x, list): return x
    if isinstance(x, str): return [x]
    return []

# --------- util p/ JSONs gerais (preparo etc.) ---------
def iter_dicts_gerais(dados):
    if isinstance(dados, dict):
        yield dados; return
    if isinstance(dados, list):
        for x in dados:
            if isinstance(x, dict):
                yield x
            elif isinstance(x, list):
                for y in x:
                    if isinstance(y, dict):
                        yield y

# --------- detectar se é arquivo de convênio ---------
def detectar_arquivo_convenio(nome_arquivo: str, dados) -> bool:
    nf = (nome_arquivo or "").lower()
    if any(k in nf for k in ["conven", "convênio", "convenio", "regras_convenio", "regrasdeconvenio"]):
        return True

    def contem_modalidades(x: dict) -> bool:
        return isinstance(x, dict) and (
            "modalidades" in x or "modalidade" in x or
            ("convenio" in x and ("modalidades" in x or "modalidade" in x))
        )

    if isinstance(dados, dict):
        if any(contem_modalidades(v) for v in dados.values()):
            return True
        for k in ["convenios", "dados", "regras", "regras_convenios", "data"]:
            if k in dados and isinstance(dados[k], (dict, list)):
                alvo = dados[k]
                if (isinstance(alvo, dict) and any(contem_modalidades(v) for v in alvo.values())) or \
                   (isinstance(alvo, list) and any(isinstance(i, dict) and contem_modalidades(i) for i in alvo)):
                    return True

    if isinstance(dados, list):
        # lista achatada (cada item tem convenio/modalidade)
        if any(isinstance(i, dict) and "convenio" in i and "modalidade" in i for i in dados):
            return True
        # lista dentro de lista (seu caso no final do arquivo)
        for sub in dados:
            if isinstance(sub, list) and any(isinstance(i, dict) and "convenio" in i and "modalidade" in i for i in sub):
                return True
        # às vezes vem um dict por convênio dentro da lista
        if any(isinstance(i, dict) and "modalidades" in i for i in dados):
            return True

    return False

# --------- indexação de convênios ---------
def indexar_convenios(dados, caminho_nome: str, documentos, contadores):
    def add_doc(conv, mod, info):
        conv_norm = str(conv).strip().lower()
        mod_norm = norm_modalidade(str(mod))
        cobre = _to_bool_cobre(info.get("cobre"))
        excecoes = _as_list(info.get("excecoes"))
        # aceita observacao singular
        observacoes = _as_list(info.get("observacoes"))
        if not observacoes:
            observacoes = _as_list(info.get("observacao"))
        sinonimos = _as_list(info.get("sinonimos"))

        page = (
            "TIPO: COBERTURA DE CONVÊNIO\n"
            f"Convênio: {conv}\n"
            f"Modalidade: {mod_norm}\n"
            f"Cobertura: {'SIM' if cobre else 'NÃO' if cobre is not None else 'indefinido'}\n"
            f"Exceções: {', '.join(map(str, excecoes)) if excecoes else '—'}\n"
            f"Observações: {', '.join(map(str, observacoes)) if observacoes else '—'}\n"
            f"Sinônimos_modalidade: {', '.join(s for s in sinonimos)}\n"
            "Palavras-chave: cobertura de convênio, cobre, autorizado, autorização, plano de saúde, aceita\n"
        )

        documentos.append(
            Document(
                page_content=page,
                metadata={
                    "tipo": "convenio",
                    "convenio": conv_norm,
                    "modalidade": (mod_norm or "").lower(),
                    "origem": caminho_nome,
                },
            )
        )
        contadores["convenio"] += 1

    # A) lista achatada (e/ou lista dentro de lista)
    if isinstance(dados, list):
        def feed(reg):
            if not isinstance(reg, dict): return
            if "convenio" in reg and "modalidade" in reg:
                reg = dict(reg)
                if "cobre" in reg:
                    reg["cobre"] = _to_bool_cobre(reg.get("cobre"))
                if "observacao" in reg and "observacoes" not in reg:
                    reg["observacoes"] = [reg["observacao"]]
                add_doc(reg["convenio"], reg["modalidade"], reg)

        for item in dados:
            if isinstance(item, dict):
                feed(item)
            elif isinstance(item, list):
                for sub in item:
                    feed(sub)
        return

    # B) dicionário por convênio ({"Unimed": {"modalidades": {...}}})
    if isinstance(dados, dict):
        # wrappers comuns
        for wrap_key in ["convenios", "dados", "regras", "regras_convenios", "data"]:
            if wrap_key in dados and isinstance(dados[wrap_key], (dict, list)):
                return indexar_convenios(dados[wrap_key], caminho_nome, documentos, contadores)

        for conv, info in dados.items():
            if not isinstance(info, dict): continue
            modalidades = info.get("modalidades") or info.get("modalidade") or {}
            if isinstance(modalidades, str): modalidades = {modalidades: info}
            if not isinstance(modalidades, dict): continue

            for mod, dados_mod in modalidades.items():
                if isinstance(dados_mod, dict):
                    dm = dict(dados_mod)
                    if "cobre" in dm: dm["cobre"] = _to_bool_cobre(dm.get("cobre"))
                    if "observacao" in dm and "observacoes" not in dm:
                        dm["observacoes"] = [dm["observacao"]]
                    add_doc(conv, mod, dm)
        return

# --------- indexação de outros JSONs (preparo etc.) ---------
def indexar_outros(dados, caminho_nome: str, documentos, contadores):
    for item in iter_dicts_gerais(dados):
        if not isinstance(item, dict): continue

        nome_exame = str(item.get("exame", "")).strip()
        if not nome_exame and not any(k in item for k in ["preparo", "sinonimos", "observacoes_adicionais", "orientacoes", "orientacoes_crianca"]):
            continue

        valor_bruto = item.get("preparo", [])
        if valor_bruto is None:
            textos_preparo = []
        elif isinstance(valor_bruto, str):
            textos_preparo = [valor_bruto]
        elif isinstance(valor_bruto, list):
            textos_preparo = valor_bruto
        else:
            textos_preparo = []

        sinonimos = _as_list(item.get("sinonimos"))
        obs_add = item.get("observacoes_adicionais", {}) or {}
        orient_crianca = _as_list(obs_add.get("orientacoes_crianca")) or _as_list(item.get("orientacoes_crianca"))
        orientacoes = _as_list(item.get("orientacoes"))

        codigo = str(item.get("codigo", "") or "")
        medicos = item.get("medicos_envolvidos", []) or item.get("medicos", [])
        if isinstance(medicos, str): medicos = [medicos]
        medicos_str = f"Médicos envolvidos: {', '.join(medicos)}" if medicos else ""

        blocos = []
        if nome_exame: blocos.append(nome_exame)
        if codigo: blocos.append(f"Código: {codigo}")
        if medicos_str: blocos.append(medicos_str)

        texto_unificado = "\n".join(blocos + textos_preparo + orient_crianca + orientacoes + sinonimos).strip()
        if not texto_unificado: continue

        documentos.append(
            Document(
                page_content=texto_unificado,
                metadata={"origem": caminho_nome}
            )
        )
        contadores["geral"] += 1

# --------- índice raso de convênios (para consumo no perguntar_ia.py) ---------
def export_indice_convenios(documentos, out_path=INDICE_CONVENIOS_JSON):
    idx = defaultdict(set)
    for d in documentos:
        md = getattr(d, "metadata", {}) or {}
        if md.get("tipo") == "convenio":
            c = md.get("convenio")
            m = (md.get("modalidade") or "").upper()
            if c and m: idx[c].add(m)
    data = {c: {"modalidades": sorted(list(mods))} for c, mods in idx.items()}
    out = Path(out_path); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# ----------------------------- MAIN -----------------------------
def main():
    embeddings = OpenAIEmbeddings()
    documentos = []
    cont = {"convenio": 0, "geral": 0}

    pasta = Path(PASTA_JSON)
    print("📂 Lendo JSONs de:", pasta.resolve())
    arquivos = sorted(pasta.glob("*.json"))
    print("🗂️ Arquivos encontrados:", [p.name for p in arquivos])

    for caminho in arquivos:
        try:
            with caminho.open("r", encoding="utf-8") as f:
                dados = json.load(f)
        except Exception as e:
            print(f"❌ Erro ao ler {caminho.name}: {e}")
            continue

        if detectar_arquivo_convenio(caminho.name, dados):
            print(f"🔎 Detectado arquivo de convênios: {caminho.name}")
            try:
                indexar_convenios(dados, caminho.name, documentos, cont)
            except Exception as e:
                print(f"❌ Erro ao indexar convênios ({caminho.name}): {e}")
            continue

        try:
            indexar_outros(dados, caminho.name, documentos, cont)
        except Exception as e:
            print(f"❌ Erro ao indexar geral ({caminho.name}): {e}")

    print(f"✅ Total de blocos indexados: {len(documentos)}")
    print(f"🏷️  Blocos de convênios indexados: {cont['convenio']}")
    print(f"📘  Blocos gerais indexados: {cont['geral']}")

    Path(PASTA_VETOR).parent.mkdir(parents=True, exist_ok=True)
    db = FAISS.from_documents(documentos, embeddings)
    db.save_local(PASTA_VETOR)
    print(f"💾 Vetor salvo em: {PASTA_VETOR}")

    export_indice_convenios(documentos)
    print(f"🗂️  Índice de convênios salvo em: {INDICE_CONVENIOS_JSON}")

    if cont["convenio"] == 0:
        print("⚠️ Aviso: nenhum bloco de convênio foi indexado. Confira o arquivo de convênios e o detector.")

if __name__ == "__main__":
    main()
