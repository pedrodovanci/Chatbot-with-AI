# perguntar_ia.py — 100% Google (Vertex AI Search + Gemini via Vertex AI)
import os
import re as _re
import re
from typing import List, Dict, Any, Optional
from typing import Any
from dotenv import load_dotenv
from google.cloud import discoveryengine_v1 as discoveryengine
from vertexai import init as vertexai_init
from vertexai.generative_models import GenerativeModel # gemini no Vertex AI
import google.cloud.aiplatform as aiplatform
print("DEBUG aiplatform:", aiplatform.__version__)


# =========================
# Util: limpar resposta
# =========================
def limpar_resposta_ia(resposta: str) -> str:
    if not resposta:
        return "Resposta: Não encontrei essa informação com clareza no documento. Recomendo confirmar com o setor responsável."
    resposta = re.sub(r"(Resposta:\s*)+", "Resposta: ", resposta, flags=re.I)
    if not resposta.strip().lower().startswith("resposta:"):
        resposta = "Resposta: " + resposta.strip()
    else:
        corpo = resposta.strip()[9:].strip()
        resposta = "Resposta: " + corpo
    linhas = [ln.strip() for ln in resposta.splitlines() if ln.strip()]
    linhas = [ln for ln in linhas if not re.fullmatch(r"(?i)resposta:\s*código:\s*$", ln)
                               and not re.fullmatch(r"(?i)código:\s*$", ln)]
    resposta = "\n".join(linhas).strip()
    if resposta.lower().strip() in {"resposta:", "resposta: código:"} or len(resposta) <= 10:
        return "Resposta: Não encontrei essa informação com clareza no documento. Recomendo confirmar com o setor responsável."
    return resposta

# =========================
# Estrutura simples de doc
# =========================
class _Doc:
    def __init__(self, page_content: str, metadata: Dict[str, Any]):
        self.page_content = page_content
        self.metadata = metadata

# =========================
# ENV / Config
# =========================
load_dotenv(override=True)

PROJECT_ID = os.getenv("PROJECT_ID")
SEARCH_LOCATION = os.getenv("SEARCH_LOCATION", "global")
SERVING_CONFIG = os.getenv("SERVING_CONFIG")  # precisa conter /engines/
VERTEX_LOCATION = os.getenv("VERTEX_LOCATION", "us-central1")

print(f"DEBUG PROJECT_ID: {PROJECT_ID}")
print(f"DEBUG SEARCH_LOCATION: {SEARCH_LOCATION}")
print(f"DEBUG SERVING_CONFIG: {SERVING_CONFIG}")
print(f"DEBUG VERTEX_LOCATION: {VERTEX_LOCATION}")

if not PROJECT_ID or not SEARCH_LOCATION or not SERVING_CONFIG:
    raise RuntimeError("Configure PROJECT_ID, SEARCH_LOCATION e SERVING_CONFIG no .env.")
if "/engines/" not in SERVING_CONFIG:
    raise RuntimeError("SERVING_CONFIG precisa conter '/engines/' (engine do Vertex AI Search).")

# inicializa Vertex AI (Gemini)
vertexai_init(project=PROJECT_ID, location=VERTEX_LOCATION)
GEMINI_MODEL = GenerativeModel("gemini-2.5-flash-lite")
# se ainda vier 404, teste: GenerativeModel("gemini-1.0-pro-001")


# =========================
# Busca — Vertex AI Search
# =========================
def _as_text(v) -> str:
    try:
        if v is None:
            return ""
        if isinstance(v, (list, tuple)):
            return " ".join(_as_text(x) for x in v if x is not None)
        if isinstance(v, dict):
            return " ".join(_as_text(x) for x in v.values() if x is not None)
        s = str(v)
        return s if s != "None" else ""
    except Exception:
        return ""

def _safe(v):
    try:
        if v is None:
            return ""
        if isinstance(v, (list, tuple)):
            return " ".join(_safe(x) for x in v)
        if isinstance(v, dict):
            return " ".join(_safe(x) for x in v.values())
        s = str(v)
        # filtra impressões de objetos proto do Discovery Engine
        return "" if s.startswith("<proto.") or "MapComposite" in s else s
    except Exception:
        return ""

def vertex_search(query: str, top_k: int = 10, filtro: Optional[Dict[str, str]] = None) -> List[_Doc]:
    client = discoveryengine.SearchServiceClient()
    
    filtro_str = ""
    if filtro:
        parts = [f'{k}="{v}"' for k, v in filtro.items() if v]
        filtro_str = " AND ".join(parts)

    content_spec = discoveryengine.SearchRequest.ContentSearchSpec(
        snippet_spec=discoveryengine.SearchRequest.ContentSearchSpec.SnippetSpec(
            return_snippet=True
        ),
        extractive_content_spec=discoveryengine.SearchRequest.ContentSearchSpec.ExtractiveContentSpec(
            max_extractive_answer_count=3,
            max_extractive_segment_count=3,
            return_extractive_segment_score=True
        ),
        summary_spec=discoveryengine.SearchRequest.ContentSearchSpec.SummarySpec(
            summary_result_count=1,
            include_citations=False
        )
    )

    req = discoveryengine.SearchRequest(
        serving_config=SERVING_CONFIG,
        query=query,
        page_size=min(max(12, top_k), 100),
        filter=filtro_str,                # usa o filtro montado
        content_search_spec=content_spec  # habilita snippet/trechos/resumo
    )

    resp = client.search(request=req, timeout=15)

    # opcional: colocar o summary como 1º doc se vier
    sumtxt = ""
    try:
        if getattr(resp, "summary", None) and getattr(resp.summary, "summary_text", ""):
            sumtxt = resp.summary.summary_text.strip()
    except Exception:
        pass

    results: List[_Doc] = []
    if sumtxt:
        results.append(_Doc(sumtxt, {"origem": "Vertex Search (summary)"}))
    for i, r in enumerate(resp):
        if i >= top_k:
            break
        # — dentro do loop for i, r in enumerate(resp): —
        doc = r.document

        def _flatten_text(x):
            out = []
            if x is None:
                return out
            if isinstance(x, str):
                # ignora lixos proto
                if x.startswith("<proto.") or "MapComposite" in x:
                    return out
                out.append(x)
                return out
            if isinstance(x, (list, tuple)):
                for it in x:
                    out.extend(_flatten_text(it))
                return out
            if isinstance(x, dict):
                # priorize campos comuns primeiro
                prefer = ("content","text","body","answer","snippet","segment","html","description","title")
                for k in prefer:
                    if k in x: out.extend(_flatten_text(x.get(k)))
                # pegue o resto das chaves também
                for k, v in x.items():
                    if k not in prefer:
                        out.extend(_flatten_text(v))
                return out
            return out

        candidates = []

        # 1) snippet do resultado (quando tem)
        snip = getattr(r, "snippet", None)
        if snip:
            candidates.append(str(snip))

        # 2) derived_struct_data completo (muitas vezes ficam aqui as respostas/segmentos)
        if doc and getattr(doc, "derived_struct_data", None):
            candidates.extend(_flatten_text(doc.derived_struct_data))

        # 3) struct_data (content/text/body/html/etc.)
        if doc and getattr(doc, "struct_data", None):
            candidates.extend(_flatten_text(doc.struct_data))

        # normaliza/filtra
        candidates = [c.strip() for c in candidates if c and isinstance(c, str) and c.strip()]
        # escolha o melhor trecho (maior costuma ser mais completo)
        text = max(candidates, key=len, default="")
        if len(text) > 6000:
            text = text[:6000]

        md = {"origem": ""}
        if not md.get("origem"):
            md["origem"] = getattr(doc, "name", "") or "desconhecida"

        if doc and getattr(doc, "struct_data", None):
            sd = doc.struct_data
            for key in ("tipo", "modalidade", "convenio"):
                val = sd.get(key)
                if val:
                    md[key] = str(val).lower()
            for k in ("uri", "url", "link", "source", "gcs_path", "file", "path", "origem"):
                val = sd.get(k)
                if val and not md.get("origem"):
                    md["origem"] = str(val)
        print("---- DEBUG RESULT ----")
        print("SNIPPET:", _safe(getattr(r, "snippet", ""))[:300])
        if doc and getattr(doc, "derived_struct_data", None):
            ds = doc.derived_struct_data
            print("EXTRACTIVE_ANSWERS:", _safe(ds.get("extractive_answers"))[:300])
            print("EXTRACTIVE_SEGMENTS:", _safe(ds.get("extractive_segments"))[:300])
        if doc and getattr(doc, "struct_data", None):
            sd = doc.struct_data
            print("STRUCT content/text/body:", _safe(sd.get("content"))[:300], _safe(sd.get("text"))[:300], _safe(sd.get("body"))[:300])
        print("----------------------")
        results.append(_Doc(text or "", md))

    print(f"[DEBUG] vertex_search: retornou {len(results)} doc(s)")
    return results

# =========================
# Filtros de contexto
# =========================
CONVENIO_KEYWORDS = {
    "convênio","convenio","plano","cobertura","cobre","aceita","autorização","autorizacao",
    "guia","pedido autorizado","carteirinha","autorizado","autoriza"
}

CONVENIOS = {
    # existentes
    "unimed":"unimed","geap":"geap","austa":"austa","bradesco":"bradesco","notredame":"notredame","prevent":"prevent",
    # principais do seu arquivo
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

def eh_pergunta_convenio(texto: str) -> bool:
    t = texto.lower()
    if any(k in t for k in CONVENIO_KEYWORDS):
        return True
    modalidades = set(sum(MODS_MAP.values(), []))
    return any(m in t for m in modalidades)

# --- normalização e expansão automática da consulta ---
_ACENTOS = str.maketrans("áàâãéêíóôõúçÁÀÂÃÉÊÍÓÔÕÚÇ", "aaaaeeioooucAAAAEEIOOOUC")

SINONIMOS = {
    "codigo": ["código", "codigo", "cod", "cód", "tuss", "procedimento"],
    "cranio": ["crânio", "cranio", "encéfalo", "encefalo", "encefálico", "encefalico"],
    "ressonancia": ["ressonância", "ressonancia", "rm", "ressonancia magnetica", "ressonância magnética"],
    "tomografia": ["tc", "tomografia", "tomografia computadorizada"],
    "orbita": ["órbita", "orbita", "orbitário", "orbitario"],
    # adicione outros termos da sua base aqui
}

def _norm(s: str) -> str:
    return (s or "").lower().translate(_ACENTOS)

def _grupo_or(palavras):
    uniq = []
    vistos = set()
    for p in palavras:
        k = _norm(p)
        if k not in vistos:
            vistos.add(k); uniq.append(p)
    if len(uniq) == 1:
        return uniq[0]
    return "(" + " OR ".join(uniq) + ")"

def expandir_consulta(q: str) -> str:
    qn = _norm(q)
    tokens = qn.split()
    grupos = []
    for tk in tokens:
        base = None
        # mapeia token para chave de sinônimos
        for chave, lista in SINONIMOS.items():
            if tk in [_norm(x) for x in lista] or tk == chave:
                base = chave
                break
        if base:
            grupos.append(_grupo_or(SINONIMOS[base]))
        else:
            grupos.append(tk)
    # reforça alguns termos úteis (boost leve)
    reforcos = []
    if any(_norm(x) in tokens for x in ["rm","ressonancia","ressonância"]):
        reforcos.append(_grupo_or(SINONIMOS["ressonancia"]))
    if any(_norm(x) in tokens for x in ["crânio","cranio","encéfalo","encefalo"]):
        reforcos.append(_grupo_or(SINONIMOS["cranio"]))
    if any(_norm(x) in tokens for x in ["código","codigo","tuss","procedimento","cod"]):
        reforcos.append(_grupo_or(SINONIMOS["codigo"]))
    consulta = " ".join(grupos + reforcos)
    return consulta.strip()

def buscar_contexto(pergunta: str, k: int = 12) -> List[_Doc]:
    base = expandir_consulta(pergunta)
     # se for pergunta de convênio, aumenta o recall
    if eh_pergunta_convenio(pergunta):
        docs = vertex_search(base, top_k=max(k, 25))
        if docs:
            return docs
    # tentativa 1: consulta expandida
    docs = vertex_search(base, top_k=k)
    if docs:
        return docs

    # tentativa 2: heurística — se falar em código, força termos usuais
    qn = _norm(pergunta)
    extras = []
    if any(w in qn for w in ["codigo","código","cod","tuss","procedimento"]):
        extras.append(_grupo_or(SINONIMOS["codigo"]))
    if any(w in qn for w in ["rm","ressonancia","ressonância"]):
        extras.append(_grupo_or(SINONIMOS["ressonancia"]))
    if any(w in qn for w in ["cranio","crânio","encéfalo","encefalo"]):
        extras.append(_grupo_or(SINONIMOS["cranio"]))

    if extras:
        docs = vertex_search(base + " " + " ".join(extras), top_k=max(k, 16))
        if docs:
            return docs

    # tentativa 3: consulta crua (como o usuário digitou) com top_k maior
    docs = vertex_search(pergunta, top_k=max(k, 20))
    return docs or []
    

def resumo_convenio(conv: str) -> str:
    def _txt(s): return (s or "").lower()
    conv_norm = _txt(conv).strip()

    docs = vertex_search(f'{conv_norm} convenio cobertura', top_k=200)

    cobertura: Dict[str, bool] = {}
    for d in docs:
        md = d.metadata or {}
        if md.get("tipo") == "convenio" and _txt(md.get("convenio")) == conv_norm:
            mod = (md.get("modalidade") or "").upper()
            if not mod:
                continue
            txt = _txt(d.page_content)
            if any(fr in txt for fr in ["não cobre", "nao cobre", "não realiza", "nao realiza", "não autorizado", "nao autorizado"]):
                cobertura.setdefault(mod, False)
            else:
                cobertura[mod] = True  # assume que cobre se não constar negação

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

# =========================
# Prompt e geração (Gemini)
# =========================
PROMPT_BASE = """
Você é um assistente de agendamentos. Responda SOMENTE com base no CONTEXTO fornecido (não invente nada fora dele). Português do Brasil.

Formato:
- Primeira linha: resposta principal em 1 frase curta.
- Até mais 3 linhas com condições essenciais (se houver).
- Se não houver evidência clara no CONTEXTO, diga explicitamente: "Não localizado no material fornecido." (sem sugerir alternativas externas).
- Quando citar código, use: "Código TUSS: <número>".
- Quando citar convênio, use: "Sim/Não, o convênio <X> (não) cobre <Y>." e acrescente condições (ex.: sem contraste, idade mínima).

Regras específicas:
- Crianças/termos infantis: priorize orientações pediátricas do CONTEXTO. Nunca oriente medicação; diga que apenas o médico pode prescrever.
- Convênios: se houver bloco de “cobertura de convênio”, use-o. Se houver conflito entre trechos, prefira o mais específico à pergunta.
- Se múltiplos códigos/condições forem possíveis, faça UMA das opções:
  (a) pergunte UMA clarificação curta (ex.: “É com contraste?”), OU
  (b) se já houver as opções no CONTEXTO, liste-as em bullets (“- ”) com seus códigos TUSS, sem inventar nada.
- Não misture preparo/código/indicação se não forem solicitados.
- Se houver orientação de idade mínima e o caso indicar menor que o mínimo, oriente a não agendar e encaminhar para avaliação médica.
- Se a PERGUNTA citar uma região específica (ex.: tórax, abdome, pelve, crânio), responda APENAS o código dessa região (não liste outras).

Rastreabilidade:
- Na última linha, inclua: "Fonte: <Origem>" utilizando o campo "📁 Origem: …" do CONTEXTO mais pertinente.

Agora responda à PERGUNTA usando estritamente o CONTEXTO.

CONTEXTO:
{contexto}

Pergunta: {pergunta}
"""

print("[DEBUG] buscando contexto...")

def perguntar_gemini(pergunta: str, contexto: str) -> str:
    print("[DEBUG] gemini: gerando...")
    resp = GEMINI_MODEL.generate_content(
        PROMPT_BASE.format(contexto=contexto, pergunta=pergunta)
    )  # <- sem request_options
    print("[DEBUG] gemini: ok")
    texto = (resp.text or "").strip()
    return limpar_resposta_ia(texto)

# =========================
# CLI
# =========================
def _precisa_clarificar(pergunta: str) -> Optional[str]:
    p = pergunta.lower()
    # TC de abdome sem detalhes
    if ("tc" in p or "tomografia" in p) and "abdome" in p:
        tem_total = "total" in p
        tem_sup = "superior" in p
        tem_contraste = ("com contraste" in p) or ("sem contraste" in p)
        if not (tem_total or tem_sup) or not tem_contraste:
            return "tc_abdome"
    # angio-TC sem dizer arterial/venosa
    if ("angio-tc" in p or "angio tc" in p) and not any(w in p for w in ["arterial","venosa"]):
        return "angio_tc"
    return None

def _rodar_clarificacao(caso: str) -> Optional[str]:
    try:
        if caso == "tc_abdome":
            tipo = input("⚠️ TC de abdome é TOTAL ou SUPERIOR? ").strip().lower()
            contraste = input("É COM contraste ou SEM contraste? ").strip().lower()
            tipo = "total" if "tot" in tipo else ("superior" if "sup" in tipo else tipo)
            if not tipo: return None
            if "sem" in contraste: contraste = "sem contraste"
            elif "com" in contraste: contraste = "com contraste"
            else: contraste = ""
            return f"código de tc de abdome {tipo} {contraste}".strip()
        if caso == "angio_tc":
            artven = input("⚠️ Angio‑TC é ARTERIAL ou VENOSA? ").strip().lower()
            if "art" in artven: return "código de angio-tc arterial"
            if "ven" in artven: return "código de angio-tc venosa"
            return None
    except KeyboardInterrupt:
        return None
    return None

if __name__ == "__main__":

    while True:
        pergunta = input("\n📩 Sua pergunta (ou 'sair'): ").strip()
        if pergunta.lower() == "sair":
            break

        pergunta_lower = pergunta.lower()
        conv_canon = next((v.lower() for k, v in CONVENIOS.items() if k in pergunta_lower), None)
        mod_canon = next((canon.lower() for alias, canon in MOD_LOOKUP.items() if alias in pergunta_lower), None)

        # Clarificação rápida para casos ambíguos
        caso = _precisa_clarificar(pergunta)
        if caso:
            nova = _rodar_clarificacao(caso)
            if nova:
                pergunta = nova
        k = 6 if any(p in pergunta_lower for p in [" e ", " com ", " e/ou ", "restrição", "acima de", "abaixo de"]) else 2
        documentos = buscar_contexto(pergunta, k=k)

        # reforço: se vier convênio + modalidade, prioriza convênio
        if conv_canon and mod_canon:
            querys = [
                f"{conv_canon} {mod_canon} cobertura de convênio",
                f"Convênio {conv_canon} cobre {mod_canon}",
                f"{conv_canon} {mod_canon} autorização plano de saúde",
                f"{conv_canon} {mod_canon} cobre",
            ]
            candidatos: List[_Doc] = []
            for q in querys:
                candidatos.extend(vertex_search(q, top_k=10))
            # remove duplicados (texto + origem)
            vistos = set()
            uniq: List[_Doc] = []
            for c in candidatos:
                chave = (c.page_content, c.metadata.get("origem", ""))
                if chave not in vistos:
                    vistos.add(chave)
                    uniq.append(c)

            def ok(doc: _Doc, only_conv=None, only_mod=None):
                md = doc.metadata or {}
                if md.get("tipo") != "convenio":
                    return False
                if only_conv and (md.get("convenio","").lower() != only_conv.lower()):
                    return False
                if only_mod and (md.get("modalidade","").lower() != only_mod.lower()):
                    return False
                return True

            docs_cov = [d for d in uniq if ok(d, conv_canon, mod_canon)] or \
                       [d for d in uniq if ok(d, only_conv=conv_canon)] or \
                       [d for d in uniq if ok(d, only_mod=mod_canon)]
            if not docs_cov:
                grandes = vertex_search(f"{conv_canon} {mod_canon}", top_k=30)
                docs_cov = [d for d in grandes if ok(d)]
            if docs_cov:
                documentos = docs_cov + [d for d in documentos if d not in docs_cov]

        contexto = "\n\n".join([
            f"📁 Origem: {doc.metadata.get('origem', 'desconhecida')}\n{doc.page_content}"
            for doc in documentos
        ])

        print("\n📄 Contexto retornado para a IA:\n")
        print(contexto)

        # 1) Gera a resposta
        resposta = perguntar_gemini(pergunta, contexto)

        # 2) Plano B: se o Gemini não achou, mas o contexto tem código TUSS
        if "Não localizado" in resposta:
            m = _re.search(r"\b(41\d{5})\b", contexto)  # ex.: 41xxxxx
            if m:
                resposta = f"Resposta: Código TUSS: {m.group(1)}.\nFonte: Vertex Search (contexto)"

        # 3) Fallback de convênio: anexa um resumo se ainda não respondeu
        if "Não localizado" in resposta and any(c in pergunta_lower for c in CONVENIOS.keys()):
            conv_canon = next((v for k, v in CONVENIOS.items() if k in pergunta_lower), None)
            if conv_canon:
                resumo = resumo_convenio(conv_canon)
                resposta = (resposta + "\n\n" + resumo).strip()

        print(f"\n🤖\n{resposta}")
