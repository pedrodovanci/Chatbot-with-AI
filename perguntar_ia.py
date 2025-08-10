import os
import json
import re
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from gemini_client import ask_gemini
from typing import Dict, Any, List, Optional

# ========= Regras determinísticas: funções utilitárias =========

def parse_question(q: str) -> Dict[str, Any]:
    """Extrai apenas o que precisamos para a decisão: peso, contraste e convênio citado."""
    ql = q.lower()

    # peso em kg (ex.: "110 kg")
    peso = None
    m = re.search(r'(\d{2,3})\s*kg\b', ql)
    if m:
        try:
            peso = int(m.group(1))
        except Exception:
            peso = None

    # contraste
    contraste = bool(re.search(r'\b(contraste|contrastado)\b', ql))

    # convênio (usa o seu dicionário CONVENIOS)
    convenio_hint = None
    for k, v in CONVENIOS.items():
        if k in ql:
            convenio_hint = v.lower()
            break

    # modalidade: para este motor, focamos em TC
    modalidade = "tc" if re.search(r'\b(tc|tomografia|angio[- ]?tc)\b', ql) else None

    return {"peso": peso, "contraste": contraste, "convenio_hint": convenio_hint, "modalidade": modalidade}

def _extract_field(text: str, rotulo: str) -> str:
    # aceita "Exceções" e "Excecoes", etc.
    alt = rotulo
    if rotulo.lower() == "exceções":
        alt = "Exce(c|ç)oe?s"
    pattern = alt if "(" in alt else re.escape(rotulo)
    m = re.search(rf"{pattern}\s*:\s*(.*)", text, flags=re.IGNORECASE)
    return m.group(1).strip() if m else ""

def pick_convenio_tc(docs_convenio: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Recebe uma lista de dicts {"metadata": ..., "content": ...}
    Retorna info do convênio especificamente para modalidade TC, se houver.
    """
    for d in docs_convenio:
        meta = d.get("metadata") or {}
        if meta.get("modalidade", "").lower() == "tc":
            content = d.get("content", "") or ""
            return {
                "convenio": (meta.get("convenio") or "").lower(),
                "cobertura": _extract_field(content, "Cobertura"),
                "excecoes": _extract_field(content, "Exceções"),
                "observacoes": _extract_field(content, "Observações")
            }
    return None

def parse_regras_tc(docs_regras: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Varre conteúdos de regras técnicas (modalidade 'tc') procurando limites de peso.
    Heurística: linhas contendo 'peso' ou 'mesa' + 'kg'. Diferencia se mencionar 'contraste'.
    """
    peso_max = None
    peso_max_contraste = None
    preparo_itens = []

    for d in docs_regras:
        txt = (d.get("content") or "").splitlines()
        for line in txt:
            low = line.lower()
            m = re.search(r'(peso|mesa).*?(\d{2,3})\s*kg', low)
            if m:
                try:
                    val = int(m.group(2))
                except Exception:
                    val = None
                if val:
                    if "contraste" in low:
                        peso_max_contraste = max(peso_max_contraste or 0, val)
                    else:
                        peso_max = max(peso_max or 0, val)
            if "preparo" in low or "orienta" in low or "jejum" in low:
                preparo_itens.append(line.strip())

    return {"peso_max": peso_max, "peso_max_contraste": peso_max_contraste, "preparo": preparo_itens}

def decide_agendamento(ent: Dict[str, Any], conv_info: Optional[Dict[str, Any]], regras: Dict[str, Any]) -> Dict[str, Any]:
    """
    Motor de decisão:
    - Cobertura NÃO -> negado_convênio
    - Cobertura INDEFINIDO -> pendente_autorização (nunca aprova)
    - Cobertura SIM -> checa regra técnica (peso). Se exceder -> negado_técnico; senão -> aprovado
    """
    if not conv_info:
        return {"status": "faltam_dados", "motivo": "Não encontrei regra do convênio para TC."}

    cobertura_raw = (conv_info.get("cobertura") or "").strip().lower()
    if "sim" in cobertura_raw:
        cobertura = "SIM"
    elif ("não" in cobertura_raw) or ("nao" in cobertura_raw):
        cobertura = "NÃO"
    else:
        cobertura = "INDEFINIDO"

    if cobertura == "NÃO":
        return {"status": "negado_convênio", "cobertura": cobertura, "convenio": conv_info.get("convenio")}

    # Checagem técnica (peso)
    peso = ent.get("peso")
    usar_contraste = ent.get("contraste", False)
    limite = None
    if usar_contraste and (regras.get("peso_max_contraste") is not None):
        limite = regras["peso_max_contraste"]
    else:
        limite = regras.get("peso_max")

    if (peso is not None) and (limite is not None) and (peso > limite):
        return {"status": "negado_técnico", "cobertura": cobertura, "limite_peso": limite, "peso_paciente": peso}

    sem_regra_peso = (peso is not None) and (limite is None)

    if cobertura == "INDEFINIDO":
        return {"status": "pendente_autorização", "cobertura": cobertura, "sem_regra_peso": sem_regra_peso}

    # cobertura == SIM
    pend_aut = "autoriz" in (conv_info.get("observacoes", "").lower())
    return {
        "status": "aprovado",
        "cobertura": cobertura,
        "pendente_autorizacao": pend_aut,
        "sem_regra_peso": sem_regra_peso
    }

def formatar_veredito(ver: Dict[str, Any], ent: Dict[str, Any], conv_info: Optional[Dict[str, Any]], regras: Dict[str, Any]) -> str:
    """Gera mensagem final coerente com a política de decisão acima."""
    if ver["status"] == "negado_convênio":
        return ("Resposta: Não é possível agendar pelo convênio (cobertura: NÃO). "
                "Alternativas: particular ou confirmar a política com o convênio.")

    if ver["status"] == "negado_técnico":
        return (f"Resposta: Não é possível realizar TC por limite técnico do equipamento: "
                f"peso do paciente {ver['peso_paciente']} kg > limite {ver['limite_peso']} kg.")

    if ver["status"] == "pendente_autorização":
        base = "Resposta: Cobertura INDEFINIDA — necessário confirmar com o convênio antes de agendar"
        if ver.get("sem_regra_peso"):
            base += "; não encontrei regra de peso nos documentos."
        return base + "."

    if ver["status"] == "aprovado":
        msg = "Resposta: Pode agendar TC pelo convênio."
        if ver.get("pendente_autorizacao"):
            msg += " Observação: exige autorização prévia."
        if ver.get("sem_regra_peso"):
            msg += " Observação: não encontrei regra de peso nos documentos."
        return msg

    # faltam_dados
    return "Resposta: Não encontrei dados suficientes para decidir. Recomendo confirmar com o setor responsável."

# Carregar .env antes de ler variáveis
load_dotenv()

# ===================== CONFIG =====================
PASTA_VETOR = "vetor/faiss_index"
ARQ_INDICE_CONVENIOS = "vetor/indice_convenios.json"
DEBUG = os.getenv("SHOW_DEBUG", "1") == "1"     # ligue/desligue prints de debug por ENV
TOP_K_BASE = int(os.getenv("TOP_K_BASE", "12")) # tuning fino do recall
# ==================================================

# ------------------- Utils limpeza ----------------
def limpar_resposta_ia(resposta: str) -> str:
    if not resposta:
        return "Resposta: Não encontrei essa informação com clareza no documento. Recomendo confirmar com o setor responsável."
    resposta = re.sub(r"(Resposta:\s*)+", "Resposta: ", resposta, flags=re.I)
    if not resposta.strip().lower().startswith("resposta:"):
        resposta = "Resposta: " + resposta.strip()
    else:
        corpo = resposta.strip()[9:].strip()
        resposta = "Resposta: " + corpo
    linhas = [ln.strip() for ln in resposta.splitlines()]
    linhas = [ln for ln in linhas if ln]
    linhas = [ln for ln in linhas if not re.fullmatch(r"(?i)resposta:\s*código:\s*$", ln)
                               and not re.fullmatch(r"(?i)código:\s*$", ln)]
    resposta = "\n".join(linhas).strip()
    if resposta.lower().strip() in {"resposta:", "resposta: código:"} or len(resposta) <= 10:
        return "Resposta: Não encontrei essa informação com clareza no documento. Recomendo confirmar com o setor responsável."
    return resposta

# ----------------- Carregamentos base --------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

embeddings = OpenAIEmbeddings()
db = FAISS.load_local(PASTA_VETOR, embeddings, allow_dangerous_deserialization=True)

# índice raso de convênios (opcional, para pré-filtro)
INDICE_CONVENIOS = {}
try:
    with open(ARQ_INDICE_CONVENIOS, "r", encoding="utf-8") as f:
        INDICE_CONVENIOS = json.load(f) or {}
except Exception:
    INDICE_CONVENIOS = {}

# ----------------- Normalizações -------------------
CONVENIOS = {
    "unimed":"unimed","geap":"geap","austa":"austa","bradesco":"bradesco","notredame":"notredame","prevent":"prevent",
    "amafresp":"amafresp","amil":"amil/golden cross","golden cross":"amil/golden cross","assomim":"assomim",
    "apas andradina":"apas andradina","apas aracatuba":"apas aracatuba","apas araçatuba":"apas aracatuba",
    "apas barretos":"apas barretos","apas fernandopolis":"apas fernandopolis","fernandópolis":"apas fernandopolis",
    "aresp":"aresp","assefaz":"assefaz","austa ocupacional":"austa ocupacional",
    "cabesp":"cabesp","cassi":"cassi","cemerp":"cemerp - probem","probem":"cemerp - probem",
    "clinica bady":"clinica bady","clínica bady":"clinica bady","dr para todos":"clinica dr para todos",
    "convida":"convida","economus":"economus","frutal licitação":"frutal licitação","hapvida":"hapvida",
    "hb":"hb","iamspe":"iamspe","jj medicina trabalho":"jj medicina trabalho",
    "life empresarial":"life empresarial","mednorte":"mednorte","medmais":"medmais","mediservice":"mediservice",
    "metra":"metra","mutuarias":"mutuarias","omint":"omint","padre albino":"padre albino",
    "partmed":"partmed","polimed":"polimed","postal saude":"postal saude","postal saúde":"postal saude",
    "prefeituras":"prefeituras","prever convenio":"prever convenio","prever":"prever",
    "pronto saude":"pronto saude","pronto saúde":"pronto saude","proasa":"proasa",
    "rede mil":"rede mil","sabesprev":"sabesprev/vivest","vivest":"sabesprev/vivest",
    "sansaude":"sansaude","são domingos":"são domingos","sao domingos":"são domingos",
    "sompo":"sompo saude","sompo saude":"sompo saude","sulamérica":"sulamerica","sulamerica":"sulamerica",
    "unifamilia":"unifamilia",
}

MODS_MAP = {
    "rm": ["rm","ressonancia","ressonância","ressonância magnética","ressonancia magnetica","rm de","ressonancia de","ressonância de"],
    "tc": ["tc","tomografia","tomografia computadorizada","angio tc","angio-tc","angio tc de","angio-tc de","tc (exceto angio-tc)"],
    "eeg": ["eeg","eletroencefalograma"],
    "enmg": ["enmg","eletroneuromiografia"],
    "us": ["us","usg","ultrassom","ultrassonografia","ecografia"],
    "rx": ["rx","raio-x","raio x","radiografia"],
    "polissonografia": ["polissonografia","psg","sono","exame do sono"],
    "angio-rm": ["angio rm","angio-rm","angiorm"],
    "angio-tc": ["angio tc","angio-tc","angiotc"],
    "rm de mama": ["rm de mama"],
    "rm de mastoide": ["rm de mastoide","rm de mastoíde"],
    "rm de face": ["rm de face"],
    "rm de órbita": ["rm de órbita","rm de orbita"],
}
MOD_LOOKUP = {alias: canon for canon, aliases in MODS_MAP.items() for alias in aliases}

CONVENIO_KEYWORDS = {
    "convênio","convenio","plano","cobertura","cobre","aceita","autorização","autorizacao",
    "guia","pedido autorizado","carteirinha","autorizado","autoriza"
}

# --------------- Extração de entidades ---------------
def extract_entities(pergunta: str):
    t = pergunta.lower()
    conv = None
    for k, v in CONVENIOS.items():
        if k in t:
            conv = v.lower()
            break

    mod = None
    for alias, canon in MOD_LOOKUP.items():
        if alias in t:
            mod = canon.lower()
            break

    # região / exame-alvo (tokens livres comuns)
    # exemplo: "abdome superior", "encéfalo", "joelho", etc.
    alvo_tokens = []
    # pegue grupos simples de 2-3 palavras após "rm", "tc", "ultrassom", etc.
    alvo_match = re.search(r"(rm|tc|ultra?ssom|raio-?x|eeg|enmg)\s*(de)?\s*([a-zçãâáéíóúõ ]{3,})", t)
    if alvo_match:
        alvo_tokens = alvo_match.group(3).strip().split()

    # sinais auxiliares
    contrast = ("contraste" in t) or ("sem contraste" in t) or ("com contraste" in t)
    machine_tesla = "3.0t" if "3.0t" in t or "3t" in t else ("1.5t" if "1.5t" in t or "1,5t" in t else None)
    horario = None
    hhmm = re.search(r"\b([01]?\d|2[0-3])[:h.]?([0-5]\d)\b", t)
    if hhmm:
        horario = f"{hhmm.group(1).zfill(2)}:{hhmm.group(2)}"
    idade = None
    idade_m = re.search(r"\b(\d{1,2})\s*anos?\b", t)
    if idade_m:
        try: idade = int(idade_m.group(1))
        except: idade = None

    return {
        "convenio": conv,
        "modalidade": mod,
        "alvo_tokens": alvo_tokens,
        "contrast_flag": contrast,
        "machine_tesla": machine_tesla,
        "horario": horario,
        "idade": idade,
        "pediatrico": any(p in t for p in {"criança","crianca","bebê","bebe","menor","filho","filha","agitado","pediatria"})
    }

# ----------- Busca + Pré-filtro de contexto -----------
def eh_pergunta_convenio(texto: str) -> bool:
    t = texto.lower()
    if any(k in t for k in CONVENIO_KEYWORDS):
        return True
    modalidades = {"rm","ressonância","ressonancia","tc","tomografia","eeg","enmg","rx","raio-x","ultrassom","us","polissonografia",
                   "angio rm","angio-rm","angio tc","angio-tc","rm de mama","rm de mastoide","rm de face","rm de órbita","rm de orbita"}
    return any(m in t for m in modalidades)

def buscar_contexto_base(pergunta: str, k: int):
    if eh_pergunta_convenio(pergunta):
        return db.similarity_search(pergunta, k=k, filter={"tipo": "convenio"})
    return db.similarity_search(pergunta, k=k)

def conv_cobre_modalidade(conv_norm: str, modalidade_up: str) -> bool | None:
    """Consulta rápida no índice raso; retorna True/False/None (desconhecido)."""
    if not INDICE_CONVENIOS or not conv_norm or not modalidade_up:
        return None
    bloco = INDICE_CONVENIOS.get(conv_norm, {})
    mods = set((bloco.get("modalidades") or []))
    if not mods:
        return None
    return modalidade_up in mods

def prefiltrar_documentos(docs, ent):
    """Mantém só o que tem alta chance de relevância à pergunta."""
    if not docs:
        return docs
    conv = ent["convenio"]
    mod = (ent["modalidade"] or "").lower()
    alvo = [a for a in ent["alvo_tokens"] if len(a) > 2]

    filtrados = []
    for d in docs:
        md = d.metadata or {}
        txt = (d.page_content or "").lower()

        # 1) se veio como 'convenio', preserve apenas os que casem convênio e (se houver) modalidade
        if md.get("tipo") == "convenio":
            if conv and md.get("convenio","") != conv:
                continue
            if mod and md.get("modalidade","") != mod:
                # Se não casou a modalidade, ainda podemos manter se a pergunta era vaga de convênio
                if eh_pergunta_convenio("convênio"):  # heurística mínima
                    pass
                else:
                    continue
            filtrados.append(d)
            continue

        # 2) para documentos gerais: exigir que contenham a modalidade e, se existir, alguma palavra do alvo
        if mod and mod not in txt:
            # alguns documentos usam "RM - CRÂNIO" etc; tente detectar " rm " como fallback
            if f" {mod} " not in txt and not txt.startswith(mod):
                continue

        if alvo:
            if not any(a in txt for a in alvo):
                # Deixa passar alguns genéricos quando a pergunta for muito vaga
                if len(alvo) >= 1:
                    continue

        filtrados.append(d)

    # fallback se filtrou demais
    return filtrados if filtrados else docs

# ---------------------- Debug helper ----------------------
def debug_print_docs(title, documentos):
    if not DEBUG:
        return
    print(f"\n===== {title} (total={len(documentos)}) =====")
    for i, doc in enumerate(documentos, 1):
        print(f"\n🔹 Documento #{i}")
        print(f"📁 Origem: {doc.metadata.get('origem', 'desconhecida')}")
        print(f"📑 Metadata: {doc.metadata}")
        print(f"📄 Conteúdo:\n{doc.page_content}")

# ----------------- Resumo convênio (existente) -----------------
def resumo_convenio(conv: str) -> str:
    conv_norm = conv.lower().strip()
    docs = db.similarity_search(conv_norm, k=200, filter={"tipo": "convenio", "convenio": conv_norm})
    cobertura = {}
    for d in docs:
        mod = (d.metadata or {}).get("modalidade", "").upper()
        txt = (d.page_content or "").lower()
        if not mod:
            continue
        if "cobertura: sim" in txt:
            cobertura[mod] = True
        elif "cobertura: não" in txt or "cobertura: nao" in txt:
            cobertura.setdefault(mod, False)

    ordem = ["RM","TC","ANGIO-RM","ANGIO-TC","TC (EXCETO ANGIO-TC)","EEG","ENMG","RX","US","POLISSONOGRAFIA",
             "RM DE MAMA","RM DE FACE","RM DE ÓRBITA","RM DE MASTOIDE"]
    cobertas = [m for m in ordem if cobertura.get(m) is True]
    nao_cobertas = [m for m in ordem if cobertura.get(m) is False]

    return (
        "RESUMO_CONVENIO\n"
        f"Convênio: {conv.upper()}\n"
        f"Cobre: {', '.join(cobertas) if cobertas else '—'}\n"
        f"Não cobre: {', '.join(nao_cobertas) if nao_cobertas else '—'}\n"
    )

# ----------------------- Modelo LLM -----------------------



# ====================== Loop CLI =========================
while True:
    pergunta = input("\n📩 Sua pergunta (ou 'sair'): ").strip()
    if pergunta.lower() == "sair":
        break

    ent = extract_entities(pergunta)

    # tamanho do contexto (perguntas mais complexas => k maior)
    k = TOP_K_BASE if any(p in pergunta.lower() for p in [" e ", " com ", " e/ou ", "restrição", "acima de", "abaixo de"]) else max(6, TOP_K_BASE // 2)

    # 1) busca bruta
    docs_base = buscar_contexto_base(pergunta, k=k)
    debug_print_docs("Contexto bruto (pós-busca)", docs_base)

    # 2) se convênio + modalidade estiverem presentes, tente reforçar com buscas focadas
    if ent["convenio"] and ent["modalidade"]:
        q_boosts = [
            f"{ent['convenio']} {ent['modalidade']} cobertura de convênio",
            f"Convênio {ent['convenio']} cobre {ent['modalidade']}",
            f"{ent['convenio']} {ent['modalidade']} autorização plano de saúde",
            f"{ent['convenio']} {ent['modalidade']} cobre",
        ]
        cand = []
        for q in q_boosts:
            cand.extend(db.similarity_search(q, k=10))
        # dedup
        vistos = set()
        cand = [c for c in cand if id(c) not in vistos and not vistos.add(id(c))]
        # mantenha só docs de convênio
        cand = [c for c in cand if (c.metadata or {}).get("tipo") == "convenio"]
        # injeta no topo
        docs_base = cand + [d for d in docs_base if d not in cand]

    # ======================
    # #3 pré-filtro por entidades extraídas
    # ======================
    docs_filtrados = prefiltrar_documentos(docs_base, ent)
    debug_print_docs("Contexto filtrado (antes do prompt)", docs_filtrados)

    # ======================
    # 3.1 – Pós-processamento determinístico (cobertura e regras técnicas)
    # ======================

    # Aqui usamos a variável 'pergunta' do loop, não 'user_question'
    entidades = parse_question(pergunta)

    # Separa documentos de convênio
    docs_convenio = [d for d in docs_filtrados if d.metadata.get("tipo") == "convenio"]

    # Separa documentos de regras técnicas (apenas TC)
    docs_regras_tc = [
        d for d in docs_filtrados
        if d.metadata.get("modalidade", "").lower() == "tc"
        and (d.metadata.get("tipo") in {"regra_exame", "exame", "guia_exame", "orientacao", "preparo", "manual_exame"})
    ]

    # Converte para o formato aceito pelas funções auxiliares
    conv_info = pick_convenio_tc([
        {"metadata": d.metadata, "content": d.page_content} for d in docs_convenio
    ])
    regras_info = parse_regras_tc([
        {"content": d.page_content} for d in docs_regras_tc
    ])

    # Decide agendamento
    veredito = decide_agendamento(entidades, conv_info, regras_info)
    veredito_formatado = formatar_veredito(veredito, entidades, conv_info, regras_info)

    # Se o veredito indicar que não deve passar pelo LLM, responde e pula para próxima pergunta
    if veredito["status"] in ("negado_convênio", "negado_técnico", "pendente_autorização", "faltam_dados"):
        resposta_final = formatar_veredito(veredito, entidades, conv_info, regras_info)
        print(resposta_final)
        continue  # volta para o início do while

    # ======================
    # 4 – Checagem rápida no índice raso de convênios (opcional, não bloqueia)
    # ======================
    resumo = ""
    if ent["convenio"]:
        try:
            resumo = resumo_convenio(ent["convenio"])
        except Exception:
            resumo = ""

    # ======================
    # 5 – Monta contexto final com origem (para transparência no console)
    # ======================
    contexto_console = "\n\n".join([
        f"📄 Origem: {doc.metadata.get('origem', 'desconhecida')}\n{doc.page_content}"
        for doc in docs_filtrados
    ])

    if DEBUG:
        print("\n📑 Contexto que será enviado ao LLM:\n")
        print(contexto_console)

    # 6) Prompt reforçado: cruzar exame/preparo/convênio/médicos e citar todas as regras relevantes
    prompt = f"""
    Você é um **assistente de agendamentos para clínica**.

    O motor de decisão JÁ determinou o VEREDITO sobre a possibilidade de agendar. 
    Sua função é apenas **redigir a resposta final** de forma clara, usando o VEREDITO e, quando aplicável, complementando com informações do CONTEXTO.

    VEREDITO (já decidido, não altere nem reinterprete):
    {veredito_formatado} 

    CONTEXTO:
    {contexto_console}

    INSTRUÇÕES:
    1. **Não mude o veredito** — ele é a decisão final.
    2. Se o veredito permitir agendamento, liste em linhas curtas as regras relevantes encontradas no CONTEXTO, seguindo esta ordem:
    - Preparo/Orientações/Restrições (jejum, contraste, idade mínima, sedação, horários, equipamento 1.5T/3.0T).
    - Cobertura e convênio.
    - Médicos envolvidos/obrigatórios (se houver).
    3. Se alguma informação não estiver no CONTEXTO, diga explicitamente "não especificado".
    4. Seja objetivo e use no máximo 3–5 linhas além da frase inicial do veredito.
    5. **Não invente** e não traga informações fora do CONTEXTO.

    Formato da resposta:
    - Primeira linha: o veredito (como está).
    - Linhas seguintes: cada regra relevante em bullet points curtos.
    
CONTEXTO:
{contexto_console}

Pergunta do atendente: {pergunta}
"""
    msg = ask_gemini(prompt)
    resposta = limpar_resposta_ia(msg or "")
    print(f"\n🤖\n{resposta}")
