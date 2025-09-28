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


def gerar_roteiro_tema(tema: str, tipo: str, tempo: str, infos_extras: str):
    # 1. Obter e formatar o prompt
    prompt_template = get_prompt_tema(tipo)

    # 2. Criar a string final do prompt
    final_prompt = prompt_template.format(
        tema=tema,
        infos_extras=infos_extras,
        tempo=tempo,

    )
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


def gerar_roteiro_pdf(arquivo, tipo: str, tempo: int, consulta: str, infos_extras: str):

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(arquivo)
        caminho_pdf = tmp_file.name

    loader = PyPDFLoader(caminho_pdf)
    documentos = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    chunks = splitter.split_documents(documentos)
    chunks = [c for c in chunks if len(c.page_content) > 300]

    texts = [c.page_content for c in chunks]

    # campo tema vazio
    if not consulta.strip():
        # Define o limite máximo de chunks que será enviado
        limite_chunks = 20

        # Verifica se o número de chunks é menor ou igual ao limite
        if len(chunks) <= limite_chunks:
            chunks_selecionados = chunks
        else:
            chunks_selecionados = chunks[:limite_chunks]

        # Junta os textos dos chunks selecionados
        texto_base = "\n".join(chunk.page_content for chunk in chunks_selecionados)
    else:
        # Embedding da consulta
        embedding_consulta = embedding_model.embed_query(consulta)

        # Embeddings dos chunks
        embeddings_chunks = embedding_model.embed_documents(texts)

        def similaridade(v1, v2):
            return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

        similaridades = [
            (similaridade(embedding_consulta, emb), chunk)
            for emb, chunk in zip(embeddings_chunks, chunks)
        ]

        # Seleciona os 5 chunks mais relevantes
        chunks_relevantes = [chunk for _, chunk in
                             sorted(similaridades, key=lambda x: x[0], reverse=True)[:5]]
        texto_base = "\n".join(chunk.page_content for chunk in chunks_relevantes)

    prompt_template = get_prompt_pdf(tipo)

    # 2. Criar a string final do prompt
    final_prompt = prompt_template.format(
        input=texto_base,
        infos_extras=infos_extras,
        tempo=tempo,

    )
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


def get_prompt_tema(tipo_roteiro: str):
    if tipo_roteiro == "A":
        return PromptTemplate.from_template(
            """
          Você é um professor de Ensino Superior e precisa elaborar o roteiro para a construção de uma apresentação de
          slides, a apresentação deverá ter {tempo} slides.                     

                   Siga o seguinte raciocínio passo a passo:

                   1. Analise o tema fornecido abaixo.
                   2. Escolha um conceito central sobre o tema que possa ser usado para a construção da apresentação.
                   3. Elabore os tópicos para os slides de forma contextualizada e baseada nesse conceito.
                   4. Defina um nome para a apresentação com base no contexto dela.
                   5. O roteiro deve ser construído para exatamente {tempo} slides.
                   6. Na elaboração do roteiro considere também as informações abaixo:

                   {infos_extras}

                   Tema da apresentação:
                   {tema}

                   Apresente o roteiro no seguinte formato:
                  
                   **Nome da apresentação:**
                   
                   **Slide X:**
                   
                   **Tópicos do slide:** ...
                   
                   (repetir até Slide {tempo})  

            """
        )

    elif tipo_roteiro == "B":
        return PromptTemplate.from_template(
            """
          Você é um professor de Ensino Superior e precisa elaborar o roteiro para a construção de uma aula,
          a aula deverá ter {tempo} minutos.                     

                   Siga o seguinte raciocínio passo a passo:

                   1. Analise o tema fornecido abaixo.
                   2. Escolha um conceito central sobre o tema que possa ser usado para ministrar uma aula.
                   3. Elabore os tópicos para a aula de forma contextualizada e baseada nesse conceito.
                   4. Defina um nome para a aula com base no contexto dela.
                   5. O roteiro deve ser construído para exatamente {tempo} minutos de aula.
                   6. Mostrar os itens a serem trabalhados em formato de tópicos.
                   7. Na elaboração do roteiro considere também as informações abaixo:

                   {infos_extras}

                   Tema da aula:
                   {tema}

                   Apresente o roteiro no seguinte formato:
           
                   **Nome da aula:**
                   
                   **Tópico X**
                   
                   **itens a serem trabalhados:** ...
            """
        )

    elif tipo_roteiro == "C":
        return PromptTemplate.from_template(
            """
           Você é um professor de Ensino Superior e precisa elaborar o roteiro para a construção de uma vídeo-aula
          a vídeo-aula deverá ter {tempo} minutos.                     

                   Siga o seguinte raciocínio passo a passo:

                   1. Analise o tema fornecido abaixo.
                   2. Escolha um conceito central sobre o tema que possa ser usado para gravação do vídeo.
                   3. Elabore os tópicos para o vídeo de forma contextualizada e baseada nesse conceito.
                   4. Defina um nome para o vídeo com base no contexto dela.
                   5. O roteiro deve ser construído para exatamente {tempo} minutos de vídeo.
                   6. Mostrar os itens a serem trabalhados em formato de tópicos.
                   7. Na elaboração do roteiro considere também as informações abaixo:

                   {infos_extras}

                   Tema da vídeo aula:
                   {tema}

                   Apresente o roteiro no seguinte formato:
               
                   **Nome da vídeo aula:**
                   
                   **Tópico X (tempo em minutos):**
                   
                   **itens a serem trabalhados:** ...
                      """
        )

    else:
        raise ValueError("Tipo de roteiro inválido")


def get_prompt_pdf(tipo_roteiro: str):
    if tipo_roteiro == "A":
        return PromptTemplate.from_template(
            """
          Você é um professor de Ensino Superior e precisa elaborar o roteiro para a construção de uma apresentação de
          slides, a apresentação deverá ter {tempo} slides.                     

                   Siga o seguinte raciocínio passo a passo:

                   1. Analise o conteúdo fornecido abaixo e identifique os conceitos mais relevantes.
                   2. Escolha um conceito central sobre o tema que possa ser usado para a construção da apresentação.
                   3. Elabore os tópicos para os slides de forma contextualizada e baseada nesse conceito.
                   4. Defina um nome para a apresentação com base no contexto dela.
                   5. O roteiro deve ser construído para exatamente {tempo} slides.
                   6. Na elaboração do roteiro considere também as informações abaixo:

                   {infos_extras}

                   Tema da apresentação:
                   {input}

                   Apresente o roteiro no seguinte formato:
               
                   **Nome da apresentação:**
                   
                   **Slide X:**
                   
                   **Tópicos do slide:** ...
                   
                   (repetir até Slide {tempo})  

            """
        )

    elif tipo_roteiro == "B":
        return PromptTemplate.from_template(
            """
           Você é um professor de Ensino Superior e precisa elaborar o roteiro para a construção de uma aula,
           a aula deverá ter {tempo} minutos.                     

                    Siga o seguinte raciocínio passo a passo:

                    1. Analise o tema fornecido abaixo.
                    2. Escolha um conceito central sobre o tema que possa ser usado para ministrar uma aula.
                    3. Elabore os tópicos para a aula de forma contextualizada e baseada nesse conceito.
                    4. Defina um nome para a aula com base no contexto dela.
                    5. O roteiro deve ser construído para exatamente {tempo} minutos de aula.
                    6. Mostrar os itens a serem trabalhados em formato de tópicos.
                    7. Na elaboração do roteiro considere também as informações abaixo:

                    {infos_extras}

                    Tema da aula:
                    {input}

                    Apresente o roteiro no seguinte formato:
                 
                    **Nome da aula:**
                    
                    **Tópico X**
                    
                    **itens a serem trabalhados:** ...
                
             """
        )

    elif tipo_roteiro == "C":
        return PromptTemplate.from_template(
            """
           Você é um professor de Ensino Superior e precisa elaborar o roteiro para a construção de uma vídeo-aula,
          a vídeo-aula deverá ter {tempo} minutos.                     

                   Siga o seguinte raciocínio passo a passo:

                   1. Analise o conteúdo fornecido abaixo e identifique os conceitos mais relevantes
                   2. Escolha um conceito central sobre o tema que possa ser usado para gravação do vídeo.
                   3. Elabore os tópicos para o vídeo de forma contextualizada e baseada nesse conceito.
                   4. Defina um nome para o vídeo com base no contexto dela.
                   5. O roteiro deve ser construído para exatamente {tempo} minutos de vídeo.
                   6. Mostrar os itens a serem trabalhados em formato de tópicos.
                   7. Na elaboração do roteiro considere também as informações abaixo:

                   {infos_extras}

                   Tema da vídeo aula:
                   {input}

                   Apresente o roteiro no seguinte formato:
             
                   **Nome da vídeo aula:**
                   
                   **Tópico X (tempo em minutos):**
                   
                   **itens a serem trabalhados:** ...
                    """
        )

    else:
        raise ValueError("Tipo de roteiro inválido")
