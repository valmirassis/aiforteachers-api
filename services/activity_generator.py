import os
import numpy as np
import tempfile
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()

# Inicializa LLM e embeddings
llm = GoogleGenerativeAI(model="models/gemini-1.5-pro-latest", temperature=0.8)
embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-exp-03-07")


def gerar_atividade_tema(tema: str, tipo: str, quantidade: str, infos_extras: str):


    prompt = get_prompt_tema(tipo)
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(tema=tema, infos_extras=infos_extras, quantidade=quantidade)


def gerar_atividade_pdf(arquivo, tipo: str, quantidade: int, consulta: str, infos_extras: str):

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

    prompt = get_prompt_pdf(tipo)
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(input=texto_base, quantidade=quantidade, infos_extras=infos_extras)


def get_prompt_tema(tipo_atividade: str):
    if tipo_atividade == "A":
        return PromptTemplate.from_template(
            """
          Você é um professor de Ensino Superior e precisa elaborar o enunciado de {quantidade} atividade do 
          tipo Estudo de caso para avaliação.                     

                               Siga o seguinte raciocínio passo a passo:

                               1. Analise o tema fornecido abaixo.
                               2. Escolha um conceito central sobre o tema que possa ser avaliado como uma atividade do tipo estudo de caso.
                                3. Elabore um enunciado claro e contextualizado baseado nesse conceito.
                               4. O enunciado deve ter pelo menos 100 palavras.
                               5. Defina um nome para a atividade com base no contexto dela.
                               6. Elabore um padrão de resposta que será usado pelo professor para a correção da atividade.
                               7. Na elaboração considere também as informações abaixo:

                               {infos_extras}

                               Tema da atividade:
                               {tema}

                               Apresente a atividade no seguinte formato:
                               inserir uma linha horizontal
                               **Nome da atividade:**

                               **Enunciado:** ...

                               **Padrão de resposta:** ...
            """
        )

    elif tipo_atividade == "B":
        return PromptTemplate.from_template(
            """
          Você é um professor de Ensino Superior e precisa elaborar o enunciado de {quantidade} atividade do tipo 
          Quadro comparativo  para avaliação.                     

                               Siga o seguinte raciocínio passo a passo:

                               1. Analise o tema fornecido abaixo.
                               2. Escolha um conceito central sobre o tema que possa ser avaliado como uma atividade do tipo quadro comparativo.
                               3. Elabore um enunciado claro e contextualizado baseado nesse conceito.
                               4. O enunciado deve ter pelo menos 100 palavras.
                               5. Defina um nome para a atividade com base no contexto dela.
                               6. Elabore um padrão de resposta em formato de tabela que será usado pelo professor para a correção da atividade.
                               7. Na elaboração considere também as informações abaixo:

                               {infos_extras}

                               Tema da atividade:
                               {tema}

                               Apresente a atividade no seguinte formato:
                               inserir uma linha horizontal
                               **Nome da atividade:**

                               **Enunciado:** ...

                               **Padrão de resposta:** ...
                      """
        )

    elif tipo_atividade == "C":
        return PromptTemplate.from_template(
            """
          Você é um professor de Ensino Superior e precisa elaborar questões discursivas.                     

                               Siga o seguinte raciocínio passo a passo:

                               1. Analise o tema fornecido abaixo.
                               2. Escolha um conceito central sobre o tema que possa ser avaliado como uma questão discursiva.
                               3. Elabore um enunciado claro e contextualizado baseado nesse conceito.
                               4. O enunciado deve ter pelo menos 50 palavras.
                               5. Defina um nome para a atividade com base no contexto dela.
                               6. Elabore um padrão de resposta que será usado pelo professor para a correção da atividade.
                               7. Contextualize o enunciado, mas no momento de pedir a resposta para o aluno, peça apenas uma coisa.
                               8. Gere exatamente {quantidade} questões
                               9. Não informe no enunciado o tamanho da resposta do aluno
                               10 Na elaboração considere também as informações abaixo:

                               {infos_extras}
                                
                                
                               Tema da atividade:
                               {tema}

                               Apresente a atividade no seguinte formato:
                               inserir uma linha horizontal
                               **Nome da atividade:**
                                
                               **Questão X** 
                               **Enunciado:** ...

                               **Padrão de resposta:** ...
                      """
        )
    elif tipo_atividade == "D":
        return PromptTemplate.from_template(
            """
         Você é um professor de Ensino Superior e precisa elaborar o enunciado de {quantidade} atividade do tipo 
          Mapa mental  para avaliação.                     

                               Siga o seguinte raciocínio passo a passo:

                               1. Analise o tema fornecido abaixo.
                               2. Escolha um conceito central sobre o tema que possa ser avaliado como uma atividade do tipo mapa mental.
                               3. Elabore um enunciado claro e contextualizado baseado nesse conceito.
                               4. O enunciado deve ter pelo menos 100 palavras.
                               5. Defina um nome para a atividade com base no contexto dela.
                               6. Elabore um padrão de resposta em formato de tópicos que deverá aparecer no mapa mental
                               7. O padrão de resposta será usado pelo professor para a correção da atividade.
                               8. Evite copiar diretamente o conteúdo original. Não mencione nomes de documentos ou fontes no enunciado.
                               9. Na elaboração considere também as informações abaixo:

                               {infos_extras}

                               Tema da atividade:
                               {tema}

                               Apresente a atividade no seguinte formato:
                               inserir uma linha horizontal
                               **Nome da atividade:**

                               **Enunciado:** ...

                               **Padrão de resposta:** ...
                                """
        )
    else:
        raise ValueError("Tipo de atividade inválido")


def get_prompt_pdf(tipo_atividade: str):
    if tipo_atividade == "A":
        return PromptTemplate.from_template(
            """
                  Você é um professor de Ensino Superior e precisa elaborar o enunciado de {quantidade} atividade do 
                  tipo Estudo de caso para avaliação.                     

                                       Siga o seguinte raciocínio passo a passo:

                                       1. Analise o conteúdo fornecido abaixo e identifique os conceitos mais relevantes
                                       2. Escolha um conceito central sobre o tema que possa ser avaliado como uma atividade do tipo estudo de caso.
                                       3. Elabore um enunciado claro e contextualizado baseado nesse conceito.
                                       4. O enunciado deve ter pelo menos 100 palavras.
                                       5. Defina um nome para a atividade com base no contexto dela.
                                       6. Elabore um padrão de resposta que será usado pelo professor para a correção da atividade.
                                       7. Evite copiar diretamente o conteúdo original. Não mencione nomes de documentos ou fontes no enunciado.
                                       8. Na elaboração considere também as informações abaixo:
                                       
                                       {infos_extras}

                                       Conteúdo de base:
                                       {input}

                                       Apresente a atividade no seguinte formato:
                                       inserir uma linha horizontal
                                       **Nome da atividade:**

                                       **Enunciado:** ...

                                       **Padrão de resposta:** ...
                    """
        )

    elif tipo_atividade == "B":
        return PromptTemplate.from_template(
            """
          Você é um professor de Ensino Superior e precisa elaborar o enunciado de {quantidade} atividade do tipo 
          Quadro comparativo  para avaliação.                     

                               Siga o seguinte raciocínio passo a passo:

                               1. Analise o conteúdo fornecido abaixo e identifique os conceitos mais relevantes
                               2. Escolha um conceito central sobre o tema que possa ser avaliado como uma atividade do tipo quadro comparativo.
                               3. Elabore um enunciado claro e contextualizado baseado nesse conceito.
                               4. O enunciado deve ter pelo menos 100 palavras.
                               5. Defina um nome para a atividade com base no contexto dela.
                               6. Elabore um padrão de resposta em formato de tabela que será usado pelo professor para a correção da atividade.
                               7. Evite copiar diretamente o conteúdo original. Não mencione nomes de documentos ou fontes no enunciado.
                               8. Na elaboração considere também as informações abaixo:
                               
                               {infos_extras}

                               Conteúdo de base:
                                {input}
                                
                               Apresente a atividade no seguinte formato:
                               inserir uma linha horizontal
                               **Nome da atividade:**

                               **Enunciado:** ...

                               **Padrão de resposta:** ...
                      """
        )

    elif tipo_atividade == "C":
        return PromptTemplate.from_template(
            """
                    Você é um professor de Ensino Superior e precisa elaborar questões discursivas.                     

                                         Siga o seguinte raciocínio passo a passo:

                                         1. Analise o conteúdo fornecido abaixo e identifique os conceitos mais relevantes
                                         2. Escolha um conceito central sobre o tema que possa ser avaliado como uma questão discursiva.
                                         3. Elabore um enunciado claro e contextualizado baseado nesse conceito.
                                         4. O enunciado deve ter pelo menos 50 palavras.
                                         5. Defina um nome para a atividade com base no contexto dela.
                                         6. Elabore um padrão de resposta que será usado pelo professor para a correção da atividade.
                                         7. Contextualize o enunciado, mas no momento de pedir a resposta para o aluno, peça apenas uma coisa.
                                         8. Gere exatamente {quantidade} questões
                                         9. Não informe no enunciado o tamanho da resposta do aluno
                                         10. Evite copiar diretamente o conteúdo original. Não mencione nomes de documentos ou fontes no enunciado.
                                         11. Na elaboração considere também as informações abaixo:

                                         {infos_extras}
                                        
                                        
                                        Conteúdo de base:
                                         {input}
                    
                                         Apresente a atividade no seguinte formato:
                                         inserir uma linha horizontal
                                         **Nome da atividade:**

                                         **Questão X** 
                                         **Enunciado:** ...

                                         **Padrão de resposta:** ...
                                """
        )
    elif tipo_atividade == "D":
        return PromptTemplate.from_template(
            """
         Você é um professor de Ensino Superior e precisa elaborar o enunciado de {quantidade} atividade do tipo 
          Mapa mental  para avaliação.                     

                               Siga o seguinte raciocínio passo a passo:

                               1. Analise o conteúdo fornecido abaixo e identifique os conceitos mais relevantes
                               2. Escolha um conceito central sobre o tema que possa ser avaliado como uma atividade do tipo mapa mental.
                               3. Elabore um enunciado claro e contextualizado baseado nesse conceito.
                               4. O enunciado deve ter pelo menos 100 palavras.
                               5. Defina um nome para a atividade com base no contexto dela.
                               6. Elabore um padrão de resposta em formato de tópicos que deverá aparecer no mapa mental
                               7. O padrão de resposta será usado pelo professor para a correção da atividade.
                               8. Evite copiar diretamente o conteúdo original. Não mencione nomes de documentos ou fontes no enunciado.
                               9. Na elaboração considere também as informações abaixo:
                               
                               {infos_extras}

                               Conteúdo de base:
                                {input}
                                
                               Apresente a atividade no seguinte formato:
                               inserir uma linha horizontal
                               **Nome da atividade:**

                               **Enunciado:** ...

                               **Padrão de resposta:** ...
                                """
        )
    else:
        raise ValueError("Tipo de atividade inválido")
