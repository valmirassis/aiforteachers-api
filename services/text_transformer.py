import numpy as np
import os
import tempfile
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client()

LLM_GEMINI = os.getenv("LLM_GEMINI")
embedding_model = GoogleGenerativeAIEmbeddings(model=os.getenv("EMBEDDING_MODEL"))
sensitive_content = """Analise o texto fornecido e verifique se ele contém qualquer tipo de conteúdo sensível, incluindo, mas não se limitando a:
                        - conteúdo sexual explícito ou implícito;
                        - violência física, psicológica ou sexual;
                        - abuso, exploração ou assédio;
                        - linguagem sexualizada ou violenta.
                        - linguagem preconceituosa contra minorias.
                        Se qualquer conteúdo sensível for identificado, retorne APENAS a mensagem:
                           ERRO: A solicitação contém conteúdo sensível e não pode ser processada.
                           """
def transformar_texto(final_prompt):
    # 3. Chamar o SDK Nativo com a configuração de raciocínio
    try:
        response = client.models.generate_content(
            model=LLM_GEMINI,
            contents=final_prompt,
            config=types.GenerateContentConfig(

                max_output_tokens=8192,

                thinking_config=types.ThinkingConfig(
                    thinking_budget=0  # <-- DESATIVA O RACIOCÍNIO
                ),
                temperature=0.4
            )
        )
        return response.text
    except Exception as e:
        return f"Erro ao gerar conteúdo: {e}"
    # Operações
def traduzir(texto: str, tipo: str, idioma: str, infos_extras: str):
    # 1. Obter e formatar o prompt
    if infos_extras and infos_extras.strip():
        infos_extras = f"Na elaboração do roteiro considere também as informações abaixo: {infos_extras}"

    prompt_template = get_prompt(tipo)

    # 2. Criar a string final do prompt
    final_prompt = prompt_template.format(
        texto=texto,
        infos_extras=infos_extras,
        idioma=idioma,
        conteudo_sensivel=sensitive_content
    )
    return transformar_texto(final_prompt)

def reescrever(texto: str, tipo: str, tom: str, infos_extras: str):
    # 1. Obter e formatar o prompt
    if infos_extras and infos_extras.strip():
        infos_extras = f"Na elaboração do roteiro considere também as informações abaixo: {infos_extras}"

    prompt_template = get_prompt(tipo)


    # 2. Criar a string final do prompt
    final_prompt = prompt_template.format(
        texto=texto,
        infos_extras=infos_extras,
        tom=tom,
        conteudo_sensivel=sensitive_content
    )
    return transformar_texto(final_prompt)
def resumir(texto: str, tipo: str, formato: str, infos_extras: str):
    # 1. Obter e formatar o prompt
    if infos_extras != "":
        infos_extras = f"""Na reescrita considere também as informações a seguir:
                   {infos_extras}"""

    prompt_template = get_prompt(tipo)


    # 2. Criar a string final do prompt
    final_prompt = prompt_template.format(
        texto=texto,
        infos_extras=infos_extras,
        formato=formato,
        conteudo_sensivel=sensitive_content
    )
    return transformar_texto(final_prompt)
def revisar(texto: str, tipo: str, infos_extras: str):
    if infos_extras and infos_extras.strip():
        infos_extras = f"Na revisão do texto considere também as informações abaixo: {infos_extras}"
    # 1. Obter e formatar o prompt
    prompt_template = get_prompt(tipo)

    # 2. Criar a string final do prompt
    final_prompt = prompt_template.format(
        texto=texto,
        infos_extras=infos_extras,
        conteudo_sensivel=sensitive_content
    )
    return transformar_texto(final_prompt)

def expandir(texto: str, tipo: str, quantidade: int, infos_extras: str):
    if infos_extras and infos_extras.strip():
        infos_extras = f"Na expansão do texto considere também as informações abaixo: {infos_extras}"
    # 1. Obter e formatar o prompt
    prompt_template = get_prompt(tipo)

    # 2. Criar a string final do prompt
    final_prompt = prompt_template.format(
        texto=texto,
        infos_extras=infos_extras,
        quantidade=quantidade,
        conteudo_sensivel=sensitive_content
    )
    return transformar_texto(final_prompt)

def criar(tema: str, tipo: str, quantidade: int, formato: str, infos_extras: str):
    if infos_extras and infos_extras.strip():
        infos_extras = f"Na criação do roteiro considere também as informações abaixo: {infos_extras}"
    # 1. Obter e formatar o prompt
    prompt_template = get_prompt(tipo)

    # 2. Criar a string final do prompt
    final_prompt = prompt_template.format(
        texto=tema,
        infos_extras=infos_extras,
        formato=formato,
        quantidade=quantidade,
        conteudo_sensivel=sensitive_content
    )
    return transformar_texto(final_prompt)

def resumirPDF(arquivo, tipo: str, formato: str, infos_extras: str):
    if infos_extras and infos_extras.strip():
        infos_extras = f"No resumo do texto considere também as informações abaixo: {infos_extras}"
    consulta = ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(arquivo)
        caminho_pdf = tmp_file.name

    loader = PyPDFLoader(caminho_pdf)
    documentos = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    chunks = splitter.split_documents(documentos)
    chunks = [c for c in chunks if len(c.page_content) > 300]

    # Define o limite máximo de chunks que será enviado
    limite_chunks = 40

    # Verifica se o número de chunks é menor ou igual ao limite
    if len(chunks) <= limite_chunks:
        chunks_selecionados = chunks
    else:
        chunks_selecionados = chunks[:limite_chunks]

    # Junta os textos dos chunks selecionados
    texto = "\n".join(chunk.page_content for chunk in chunks_selecionados)

    prompt_template = get_prompt(tipo)

    # 2. Criar a string final do prompt
    final_prompt = prompt_template.format(
        texto=texto,
        infos_extras=infos_extras,
        formato=formato,
        conteudo_sensivel=sensitive_content
    )
    return transformar_texto(final_prompt)

def get_prompt(tipo: str):
    if tipo == "A":
        return PromptTemplate.from_template(
            """
                   Você é um tradutor e precisa traduzir o texto a seguir. 
                                                        
                   Siga o raciocínio passo a passo:
                     
                   {conteudo_sensivel}   

                   1. Analise o text fornecido a seguir.
                   2. Traduza para o idioma informado.
                   3. Mantenha o contexto do texto.
                   4. Não acrescente informações no texto.
                   4. Não exiba mensagens de saudação, apenas apresente o texto traduzido.
                   
                   {infos_extras}

                   Texto a ser traduzido
                   {texto}

                    Idioma: {idioma}
                  
                   **Texto traduzido do idioma X para o idioma Y**:
            """
        )

    elif tipo == "B":
        return PromptTemplate.from_template(
              """
                    Reescreva o texto a seguir de forma direta sem análises ou saudações.                   
                   Siga o raciocínio passo a passo:
                   {conteudo_sensivel}

                   1. Analise o texto fornecido a seguir.
                   2. Reescreva o texto com o tom {tom}
                   3. Mantenha o contexto do texto.
                   4. Não explique ou faça análises do texto, apenas reescreva de forma direta.
                  
                   {infos_extras}

                   Texto a ser reescrito:
                   {texto}

                   **Texto reescrito com o tom {tom}**:
                """
        )

    elif tipo == "C":
        return PromptTemplate.from_template(
            """
             Resuma o texto a seguir de forma direta sem análises ou saudações.                   
                   Siga o raciocínio passo a passo:
                   {conteudo_sensivel}

                   1. Analise o texto fornecido a seguir.
                   2. Resuma o texto no formato {formato}
                   3. Mantenha o contexto do texto.
                   4. Não explique ou faça análises do texto, apenas resuma de forma direta.
                  
                   {infos_extras}

                   Texto a ser resumido:
                   {texto}

                   **Texto resumido em formato {formato}**:
                """

        )
    elif tipo == "CPDF":
        return PromptTemplate.from_template(
            """
             Resuma o texto a seguir de forma direta sem análises ou saudações.                   
                   Siga o raciocínio passo a passo:
                   {conteudo_sensivel}

                   1. Analise o texto fornecido a seguir.
                   2. Resuma o texto no formato {formato}
                   3. Mantenha o contexto do texto.
                   4. Não explique ou faça análises do texto, apenas resuma de forma direta.

                   {infos_extras}

                   Texto a ser resumido:
                   {texto}

                   **Texto resumido em formato {formato}**:
                """

        )
    elif tipo == "D":
        return PromptTemplate.from_template(

            """
                          Você é um professor e precisa revisar e corrigir o texto seguir.                   
                          Siga o raciocínio passo a passo:
                          {conteudo_sensivel}
    
                          1. Analise o texto fornecido a seguir.
                          2. Reescreva o texto revisando e corrigindo.
                          4. Corrija gráfia e concordância.
                          4. Mantenha o contexto do texto.
                          5. Não acrescente informações no texto
                          6. Não exiba mensagens de saudação, apenas apresente o texto revisado e corrigido
                          
                          {infos_extras}
    
                          Texto a ser revisto e corrigido:
                          {texto}

    
                          **Texto reescrito e corrigido**:
                          **Correções e alterações efetuadas:**
    
                               """
        )
    elif tipo == "E":
        return PromptTemplate.from_template(

            """
            Você é um professor e precisa expandir o texto a seguir
                          Siga rigorosamente as instruções abaixo:
                          {conteudo_sensivel}

                        1. Analise o texto original mantendo integralmente seu contexto, tema e objetivo.
                        2. Reescreva o texto de forma expandida, acrescentando aproximadamente {quantidade} palavras em relação ao texto original.
                        3. A expansão deve ocorrer por meio de:
                           - maior detalhamento das ideias já presentes,
                           - explicitação de conceitos implícitos,
                           - inclusão de exemplos explicativos quando pertinente,
                           sem introduzir novos tópicos ou alterar o sentido do texto.
                        4. O texto final pode variar até ±10% da quantidade de palavras solicitadas.
                        5. Não repita frases do texto original de forma redundante.
                        6. Não apresente comentários, explicações, saudações ou marcações adicionais.
                      
                          {infos_extras}
                          Texto a ser expandido:
                          {texto}
                   
                          **Texto expandido**:
        
                               """
        )
    elif tipo == "F":
        return PromptTemplate.from_template(

            """
                          Você é um professor e precisa criar o texto seguir.                   
                          Siga o raciocínio passo a passo:
                          {conteudo_sensivel}

                          1. Analise o tema fornecido a seguir.
                          2. Crie um texto contextualizado sobre o tema respeitanto com aproximadamente {quantidade} de palavras
                          3. Mantenha uma estrutura organizada.
                          4. Não apresente comentários, explicações, saudações ou marcações adicionais.
                        
                          {infos_extras}

                          Tema do texto a ser criado:
                          {texto}
                          
                          Formato do texto: {formato}

                          **Texto criado**:

                               """
        )
    else:
        raise ValueError("Tipo de texto inválido")

