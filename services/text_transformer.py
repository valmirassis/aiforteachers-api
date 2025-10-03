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
    if infos_extras != "":
        infos_extras = f"""Na tradução considere também as informações a seguir:
                      {infos_extras}"""
    prompt_template = get_prompt(tipo)

    # 2. Criar a string final do prompt
    final_prompt = prompt_template.format(
        texto=texto,
        infos_extras=infos_extras,
        idioma=idioma
    )
    return transformar_texto(final_prompt)

def reescrever(texto: str, tipo: str, tom: str, infos_extras: str):
    # 1. Obter e formatar o prompt
    if infos_extras != "":
        infos_extras = f"""Na reescrita considere também as informações a seguir:
                   {infos_extras}"""

    prompt_template = get_prompt(tipo)


    # 2. Criar a string final do prompt
    final_prompt = prompt_template.format(
        texto=texto,
        infos_extras=infos_extras,
        tom=tom
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
        formato=formato
    )
    return transformar_texto(final_prompt)
def revisar(texto: str, tipo: str, infos_extras: str):
    if infos_extras != "":
        infos_extras = f"""Na revisão considere também as informações a seguir:
                      {infos_extras}"""
    # 1. Obter e formatar o prompt
    prompt_template = get_prompt(tipo)

    # 2. Criar a string final do prompt
    final_prompt = prompt_template.format(
        texto=texto,
        infos_extras=infos_extras,
    )
    return transformar_texto(final_prompt)

def expandir(texto: str, tipo: str, quantidade: int, infos_extras: str):
    if infos_extras != "":
        infos_extras = f"""Na expansão considere também as informações a seguir:
                      {infos_extras}"""
    # 1. Obter e formatar o prompt
    prompt_template = get_prompt(tipo)

    # 2. Criar a string final do prompt
    final_prompt = prompt_template.format(
        texto=texto,
        infos_extras=infos_extras,
        quantidade=quantidade
    )
    return transformar_texto(final_prompt)

def criar(tema: str, tipo: str, quantidade: int, formato: str, infos_extras: str):
    if infos_extras != "":
        infos_extras = f"""Na criação considere também as informações a seguir:
                      {infos_extras}"""
    # 1. Obter e formatar o prompt
    prompt_template = get_prompt(tipo)

    # 2. Criar a string final do prompt
    final_prompt = prompt_template.format(
        texto=tema,
        infos_extras=infos_extras,
        formato=formato,
        quantidade=quantidade
    )
    return transformar_texto(final_prompt)

def resumirPDF(arquivo, tipo: str, formato: str, infos_extras: str):
    if infos_extras != "":
        infos_extras = f"""Na criação considere também as informações a seguir:
                         {infos_extras}"""
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
        formato=formato
    )
    return transformar_texto(final_prompt)

def get_prompt(tipo: str):
    if tipo == "A":
        return PromptTemplate.from_template(
            """
                   Você é um tradutor e precisa traduzir o texto a seguir.                   
                   Siga o seguinte raciocínio passo a passo:

                   1. Analise o text fornecido a seguir.
                   2. Traduza para o idioma informado.
                   3. Mantenha o contexto do texto.
                   4. Não exiba mensagens de saudação, apenas apresente o texto
                   
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
                   Siga o seguinte raciocínio passo a passo:

                   1. Analise o text fornecido a seguir.
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
                   Siga o seguinte raciocínio passo a passo:

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
                   Siga o seguinte raciocínio passo a passo:

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
                          Siga o seguinte raciocínio passo a passo:
    
                          1. Analise o texto fornecido a seguir.
                          2. Reescreva o texto revisando e corrigindo.
                          3. Mantenha o contexto do texto.
                          4. Não exiba mensagens de saudação, apenas apresente o texto revisado e corrigido
                          
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
                          Você é um professor e precisa expandir o texto seguir.                   
                          Siga o seguinte raciocínio passo a passo:

                          1. Analise o texto fornecido a seguir.
                          2. Reescreva o texto expandindo-o, considere a quantidade de caracteres fornecida para expandir
                          3. Mantenha o contexto do texto.            
                          4. Não exiba mensagens de saudação, apenas apresente o texto expandido.
                      
                          {infos_extras}

                          Texto a ser revisto e corrigido:
                          {texto}

                          Aumente até: {quantidade} de caracteres com espaço.

                          **Texto expandido**:
        
                               """
        )
    elif tipo == "F":
        return PromptTemplate.from_template(

            """
                          Você é um professor e precisa criar o texto seguir.                   
                          Siga o seguinte raciocínio passo a passo:

                          1. Analise o tema fornecido a seguir.
                          2. Crie um texto contextualizado sobre o tema respeitanto do formato e quantidade de caracteres fornecidos.
                          3. Mantenha uma estrutura organizada.
                          4. Não exiba mensagens de saudação, apenas apresente o texto gerado
                        
                          {infos_extras}

                          Tema do texto a ser criado:
                          {texto}
                          Tamanho do texto: Até {quantidade} caracteres com espaço;
                          Formato do texto: {formato}

                          **Texto expandido**:

                               """
        )
    else:
        raise ValueError("Tipo de texto inválido")

