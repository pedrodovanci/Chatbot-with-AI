import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import re



def limpar_resposta_ia(resposta: str) -> str:
    """
    Remove duplicações de 'Resposta:' em qualquer ponto e padroniza a saída da IA.
    """
    if not resposta:
        return ""

    # Remove múltiplas ocorrências seguidas de 'Resposta:'
    resposta = resposta.replace("Resposta: Resposta:", "Resposta:")

    # Remove todas as repetições internas
    resposta = re.sub(r"(Resposta:\s*)+", "Resposta: ", resposta, flags=re.IGNORECASE)

    # Garante que comece com 'Resposta: '
    if not resposta.strip().lower().startswith("resposta:"):
        resposta = "Resposta: " + resposta.strip()
    else:
        resposta = "Resposta: " + resposta.strip()[9:].strip()

    # Remove quebras de linha duplicadas e espaços
    linhas = resposta.splitlines()
    linhas_limpa = [linha.strip() for linha in linhas if linha.strip()]
    resposta_final = "\n".join(linhas_limpa)

    return resposta_final.strip()
# Carregar variável de ambiente
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Caminhos
PASTA_VETOR = "vetor/faiss_index"

# Inicializa os embeddings e carrega o índice FAISS
embeddings = OpenAIEmbeddings()
db = FAISS.load_local(PASTA_VETOR, embeddings, allow_dangerous_deserialization=True)

# Mapas de normalização
CONVENIOS = {
    "unimed": "unimed",
    "geap": "geap",
    "austa": "austa",
    "bradesco": "bradesco",
    "notredame": "notredame",
    "prevent": "prevent",
    # adicione variações se precisar (ex.: "unimed franca": "unimed")
}

MODS_MAP = {
    # canônico -> lista de sinônimos
    "rm": ["rm", "ressonancia", "ressonância", "ressonância magnética", "ressonancia magnetica"],
    "tc": ["tc", "tomografia", "tomografia computadorizada"],
    "eeg": ["eeg", "eletroencefalograma"],
    "enmg": ["enmg"],
    "us": ["us", "usg", "ultrassom", "ultrassonografia"],
    "polissonografia": ["polissonografia", "sono"],
    "angiografia": ["angiografia", "angio"],
}
# Inverso para lookup rápido
MOD_LOOKUP = {alias: canon for canon, aliases in MODS_MAP.items() for alias in aliases}

# Inicializa o modelo de linguagem
llm = ChatOpenAI(
    model_name="gpt-4",  # ou "gpt-3.5-turbo"
    temperature=0,
    openai_api_key=OPENAI_API_KEY
)

# Loop para perguntar
# Loop para perguntar
while True:
    pergunta = input("\n📩 Sua pergunta (ou 'sair'): ").strip()
    if pergunta.lower() == "sair":
        break

    pergunta_lower = pergunta.lower()

    # Detecta convênio (pega o primeiro que aparecer)
    conv_canon = None
    for conv in CONVENIOS.keys():
        if conv in pergunta_lower:
            conv_canon = CONVENIOS[conv]
            break

    # Detecta modalidade pelo lookup de sinônimos
    mod_canon = None
    for alias, canon in MOD_LOOKUP.items():
        if alias in pergunta_lower:
            mod_canon = canon
            break

    k = 4 if any(p in pergunta_lower for p in [" e ", " com ", " e/ou ", "restrição", "acima de", "abaixo de"]) else 2

    # Buscar contexto mais relevante (default)
        # Buscar contexto mais relevante (default)
    documentos = db.similarity_search(pergunta, k=k)

    # Se identificarmos convênio e modalidade, priorize blocos de convênio via filtro manual
    if conv_canon and mod_canon:
        # 1) tentativa forte com palavras-chave
        querys = [
            f"{conv_canon} {mod_canon} cobertura de convênio",
            f"Convênio {conv_canon} cobre {mod_canon}",
            f"{conv_canon} {mod_canon} autorização plano de saúde",
            f"{conv_canon} {mod_canon} cobre",
        ]

        candidatos = []
        for q in querys:
            candidatos.extend(db.similarity_search(q, k=10))

        # remove duplicados preservando ordem
        vistos = set()
        candidatos = [c for c in candidatos if id(c) not in vistos and not vistos.add(id(c))]

        def ok(doc, only_conv=None, only_mod=None):
            md = doc.metadata or {}
            if md.get("tipo") != "convenio":
                return False
            if only_conv and md.get("convenio") != only_conv:
                return False
            if only_mod and md.get("modalidade") != only_mod:
                return False
            return True

        # 2) filtro ideal (convênio + modalidade)
        docs_cov = [d for d in candidatos if ok(d, conv_canon, mod_canon)]

        # 3) fallback por convênio
        if not docs_cov:
            docs_cov = [d for d in candidatos if ok(d, only_conv=conv_canon)]

        # 4) fallback por modalidade
        if not docs_cov:
            docs_cov = [d for d in candidatos if ok(d, only_mod=mod_canon)]

        # 5) se ainda vazio, tenta uma busca genérica grande e filtra
        if not docs_cov:
            grandes = db.similarity_search(f"{conv_canon} {mod_canon}", k=30)
            docs_cov = [d for d in grandes if ok(d)]

        # Prioriza convênios no topo
        if docs_cov:
            documentos = docs_cov + [d for d in documentos if d not in docs_cov]


    contexto = "\n\n".join([doc.page_content for doc in documentos])
    print("\n📄 Contexto retornado para a IA:\n")
    print(contexto)
    
    prompt = f"""

Quando a pergunta for sobre se um convênio atende ou não um exame/modalidade:
- Responda sempre de forma direta e clara com "Sim, o convênio X cobre Y" ou "Não, o convênio X não cobre Y".
- Se houver condições (ex.: só cobre alguns casos, só cobre exames sem contraste), mencione após a resposta direta.

Você é um assistente de agendamentos treinado para responder perguntas de funcionários de uma clínica médica com base em documentos internos relacionados a exames (como RM, TC, etc).

Responda sempre com base apenas no conteúdo fornecido no contexto.

Caso a pergunta mencione ou sugira que o paciente seja uma criança, bebê, adolescente, ou utilize termos como 'filho', 'criança', 'pequeno', 'menor', 'bebê', 'filha', 'menina', 'agitado', 'nervoso', 'não para quieto', 'não colabora' etc., busque no contexto qualquer bloco com orientações específicas para pediatria.

Se houver um trecho sobre pacientes infantis ou crianças agitadas, utilize-o na resposta com prioridade. Caso contrário, informe que o documento não possui essa informação específica.

Se a pergunta mencionar um convênio (como Unimed, GEAP, Austa, etc.) e uma modalidade (como RM, TC, EEG...), busque especificamente por blocos de **cobertura de convênio**.

Caso encontre, utilize a informação presente nesse bloco.

Se a pergunta mencionar um convênio (Unimed, GEAP, Austa, etc.), mesmo sem citar explicitamente a modalidade (ex: só dizer “ressonância” ou “exame”), e o contexto retornar blocos de cobertura, utilize esses blocos.

Nunca responda "não encontrado" se houver qualquer bloco sobre o convênio e modalidade combinados. Sempre prefira o conteúdo mais próximo, mesmo que esteja em linguagem diferente da pergunta.

Regras importantes:
- Seja direto e objetivo.
- Responda apenas o que foi perguntado.
- Nunca misture informações de preparo, código ou indicações se não forem solicitadas.
- Se houver recomendação para que o paciente procure orientação médica, informe isso claramente na resposta.
- Nunca diga que o documento não fornece informação se houver qualquer recomendação ou orientação indireta no texto.
- Se a informação não estiver clara no documento, responda educadamente que não está disponível.
- Se houver uma orientação no documento sobre a idade mínima para realizar um exame (ex: “a partir de 5 anos”), use essa informação com prioridade.
- Caso a idade do paciente seja inferior ao indicado, oriente o funcionário a não agendar e encaminhar para avaliação médica.

Regras complementares para inferências controladas:

- Se perguntado explicitamente sobre **preparo** (como jejum) e o documento não tiver essa informação clara, busque exames semelhantes no contexto fornecido:
  - Caso encontre exames semelhantes (mesma modalidade, mesmo tipo de contraste) que exijam jejum, responda dessa forma:  
    "O documento não especifica claramente o preparo para este exame, porém, exames semelhantes geralmente exigem jejum de 2 horas (sem contraste) ou 4 horas (com contraste). Recomendo confirmar com o setor responsável."
  - Se não encontrar exemplos semelhantes, informe claramente que "o documento não especifica essa informação".

- Quando perguntado sobre **contraindicações** ou **restrições** específicas e isso não estiver claro, responda da seguinte forma:
  - "O documento não menciona restrições específicas para este exame. No entanto, se o paciente possui histórico clínico relevante ou dúvidas sobre o procedimento, é recomendável confirmar diretamente com a equipe médica antes do agendamento."

- Quando questionado sobre situações especiais (gestantes, pacientes renais, alérgicos a contraste), caso o documento não mencione explicitamente, responda:
  - "O documento não fornece informações específicas sobre esta situação. Recomendo verificar com o médico solicitante ou com o setor de agendamento."

Nunca responda com certezas absolutas em situações não descritas claramente no documento.

Exemplo:
Pergunta: Qual o preparo para a RM do encéfalo?
Resposta: Jejum de 2 horas. O exame é realizado em 1,5T, 3,0T ou campo aberto com restrições. Verificar se o paciente obeso (>100kg) cabe na bobina. Agendar em horário branco ou neuro.

Interprete o campo `medicos_envolvidos` da seguinte forma:

1. Quando presente, representa a lista de médicos obrigatórios para realização do exame.
2. Quando ausente, o exame pode ser agendado com qualquer médico, exceto se o texto do exame mencionar explicitamente algum nome.
3. Nunca crie nomes de médicos por dedução. Só mencione nomes reais se estiverem:
   - No campo `medicos_envolvidos`, ou
   - Especificados no texto do preparo, observações ou instruções do exame.

Se a pergunta for sobre “quem pode realizar o exame”, e não houver informação, retorne que o dado não está disponível.

Lembre-se: muitos usuários podem fazer perguntas com linguagem informal, como "meu filho é agitado", "posso dar calmante?", "precisa de sedativo?", etc. Sempre que houver dúvidas relacionadas a crianças agitadas ou medicação, a resposta correta é: a clínica **não orienta o uso de medicamentos sem avaliação médica**. Apenas o médico pode prescrever calmantes ou sedativos. Utilize os documentos disponíveis para oferecer uma orientação clara e responsável ao usuário.

Considere variações semânticas nos termos infantis, como: 'pequeno', 'idade', 'pode fazer com 4 anos?', 'só a partir de X anos', 'criança menor', 'abaixo de X anos', etc. Sempre que possível, busque o trecho mais relacionado à idade mínima permitida nos documentos.

Dê preferência a qualquer bloco que utilize termos semelhantes ou equivalentes ao da pergunta, mesmo que a linguagem não seja idêntica.

Você pode usar o contexto abaixo para responder:


{contexto}

🔎 Pergunta: {pergunta}
""".replace("{contexto}", contexto).replace("{pergunta}", pergunta)

    resposta = llm.predict(prompt)
    resposta = limpar_resposta_ia(resposta)
    print(f"\n🤖\n{resposta}")

