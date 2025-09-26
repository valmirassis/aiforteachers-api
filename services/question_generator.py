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
llm = GoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0.8)
embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")


def gerar_questoes_tema(tema: str, tipo: str, qtd: int, dificuldade: str):
    info_dificuldade = ""
    if dificuldade.lower() == "sim":
        info_dificuldade = (
            "Para cada questão gerada, gere uma versão com nível de dificuldade fácil "
            "usando taxonomia de Bloom níveis 1 ou 2 (lembrança ou compreensão) "
            "e uma versão difícil usando taxonomia de Bloom níveis 3, 4 ou 5 (aplicação, análise ou avaliação)."
        )

    prompt = get_prompt_tema(tipo)
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(input=tema, quantidade=qtd, dificuldade=info_dificuldade)


def gerar_questoes_pdf(arquivo, tipo: str, qtd: int, consulta: str, dificuldade: str):
    info_dificuldade = ""
    if dificuldade.lower() == "sim":
        info_dificuldade = (
            "Para cada questão gerada, gere uma versão com nível de dificuldade fácil "
            "usando taxonomia de Bloom níveis 1 ou 2 (lembrança ou compreensão) "
            "e uma versão difícil usando taxonomia de Bloom níveis 3, 4 ou 5 (aplicação, análise ou avaliação)."
        )

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
    return chain.run(input=texto_base, quantidade=qtd, dificuldade=info_dificuldade)


def get_prompt_tema(tipo_questao: str):
    if tipo_questao == "A":
        return PromptTemplate.from_template(
            """
        Você é um professor de Ensino Superior e precisa elaborar {quantidade} questão(ões)
                      de múltipla escolha para avaliação.                     
                    
                     Siga o seguinte raciocínio passo a passo:

                     1. Analise o tema fornecido abaixo.
                     2. Escolha um conceito central sobre o tema que possa ser avaliado com uma pergunta objetiva.
                     3. Elabore um enunciado claro, contextualizado, com pelo menos 50 palavras, baseado nesse conceito.
                     4. Verifique se a **resposta correta não está explícita no enunciado**.
                     5. Crie 5 alternativas (a–e), sendo apenas uma correta.
                     6. Indique qual alternativa é a correta.
                     7. Justifique por que a alternativa correta está certa.
                     8. Justifique por que as demais alternativas estão incorretas.
                     9. Apresente quebras de linha entre as alternativas e entre as justificativas para ficar bem separado.
                     
                     {dificuldade}
                     
                     Tema da questão:
                     {input}

                     Apresente a questão no seguinte formato (cada alternativa deve estar em uma linha separada):
                     inserir uma linha horizontal
                     **Questão X**
                     
                     **Enunciado:** ...
                     
                     Sobre o exposto acima assinale a alternativa CORRETA: 
                     
                     a) [Elabore a primeira alternativa e insira uma quebra de linha.]
                     
                     b) [Elabore a segunda alternativa e insira uma quebra de linha.]
                     
                     c) [Elabore a terceira alternativa e insira uma quebra de linha.]
                     
                     d) [Elabore a quarta alternativa e insira uma quebra de linha.]
                     
                     e) [Elabore a quinta alternativa e insira uma quebra de linha.]                   
                     
                     
                     **Alternativa correta:** ...
                     
                     **Justificativa (raciocínio passo a passo):**
                     
                     Confirme se realmente não é possível responder a questão com as informações presentes no enunciado
                     se sim, gere outra questão.
            """
        )

    elif tipo_questao == "B":
        return PromptTemplate.from_template(
            """
         Você é um professor de Ensino Superior e precisa elaborar exatamente {quantidade} questão(ões) 
                        de múltipla escolha com afirmativas.
                        
                        Siga o seguinte raciocínio passo a passo:
                        
                        1. Analise o tema fornecido e identifique 4 ideias ou conceitos distintos sobre o tema que podem ser transformados em afirmativas.
                        2. Elabore um enunciado claro e contextualizado com pelo menos 50 palavras.
                        3. Escreva 4 afirmativas numeradas de I a IV.
                        4. Elabore 5 alternativas (a–e), com diferentes combinações de afirmativas corretas.
                        5. Verifique se **as afirmativas ou o enunciado não indicam claramente qual alternativa é correta apenas pela redação**.
                        6. Indique qual alternativa é a correta.
                        7. Justifique de forma detalhada por que a alternativa correta está certa.
                        8. Justifique de forma detalhada por que as demais alternativas estão incorretas.
                        9. Apresente quebras de linha entre as alternativas e entre as justificativas para ficar bem separado. 
                        
                        {dificuldade}
                        
                        Tema da questão:
                        {input}
                        
                        Apresente a questão no seguinte formato (cada alternativa deve estar em uma linha separada):
                        inserir uma linha horizontal
                        **Questão X**
                        **Enunciado:** ...
                        
                        I. [Elabore a primeira afirmativa e insira uma quebra de linha.]
                        
                        II. [Elabore a segunda afirmativa e insira uma quebra de linha.]
                        
                        III. [Elabore a terceira afirmativa e insira uma quebra de linha.]
                        
                        IV. [Elabore a quarta afirmativa e insira uma quebra de linha.]
                        
                        Assinale a alternativa que apresenta todas as afirmativas corretas.
                        
                        a) [Elabore a primeira alternativa e insira uma quebra de linha.]
                                                
                        b) [Elabore a segunda alternativa e insira uma quebra de linha.]
                        
                        c) [Elabore a terceira alternativa e insira uma quebra de linha.]
                        
                        d) [Elabore a quarta alternativa e insira uma quebra de linha.]
                        
                        e) [Elabore a quinta alternativa e insira uma quebra de linha.]
                        
                        
                        **Alternativa correta:** ...
                        
                        **Justificativa (raciocínio passo a passo):**
                        
                        Confirme se realmente não é possível responder a questão com as informações presentes no enunciado
                        se sim, gere outra questão.
            """
        )

    elif tipo_questao == "C":
        return PromptTemplate.from_template(
            """
             Você é um professor de Ensino Superior e precisa elaborar {quantidade} questão(ões) 
                        do tipo múltipla escolha com estrutura de asserção e razão.
                        
                        Siga o seguinte raciocínio passo a passo:
                        
                        1. Analise o tema fornecido e selecione um conceito principal.
                        2. Elabore um enunciado introdutório (contexto) com pelo menos 50 palavras.
                        3. Crie uma **asserção (I)** e uma **razão (II)** relacionadas ao conceito.
                        4. Elabore 5 alternativas (a–e) sobre a veracidade das asserções e se a razão justifica a asserção.
                        5. Verifique se **o enunciado não antecipa a resposta correta nem contém pistas óbvias**.
                        6. Indique qual alternativa é a correta.
                        7. Justifique de forma detalhada por que a alternativa correta está certa.
                        8. Justifique de forma detalhada por que as demais alternativas estão incorretas.
                        9. Apresente quebras de linha entre as alternativas e entre as justificativas para ficar bem separado. 
                      
                        {dificuldade}
                        Tema da questão:
                        {input}
                        
                        Apresente a questão no seguinte formato (cada alternativa deve estar em uma linha separada):
                        inserir uma linha horizontal
                        **Questão X**
                        **Enunciado:** ...
                        
                        Considerando as informações, avalie as asserções a seguir e a relação proposta entre elas.
                        
                        I. ... 
                        
                        PORQUE  
                        
                        II. ... 
                        
                        A respeito dessas asserções, assinale a opção CORRETA. 
                        
                        a) As asserções I e II são proposições verdadeiras, e a II é uma justificativa correta da I.
                        
                        b) As asserções I e II são proposições verdadeiras, mas a II não é uma justificativa correta da I.
                        
                        c) A asserção I é uma proposição verdadeira, e a II é uma proposição falsa.
                        
                        d) A asserção I é uma proposição falsa, e a II é uma proposição verdadeira.
                        
                        e) As asserções I e II são proposições falsas.
                        
                        **Alternativa correta:** ...
                        
                        **Justificativa (raciocínio passo a passo):**
                        
                        Confirme se realmente não é possível responder a questão com as informações presentes no enunciado
                        se sim, gere outra questão.
            """
        )

    else:
        raise ValueError("Tipo de questão inválido")


def get_prompt_pdf(tipo_questao: str):
    if tipo_questao == "A":
        return PromptTemplate.from_template(
               """
                     Você é um professor de Ensino Superior e precisa elaborar {quantidade} questão(ões)
                      de múltipla escolha para avaliação.                     
                    
                     Siga o seguinte raciocínio passo a passo:

                     1. Analise o conteúdo fornecido abaixo e identifique os conceitos mais relevantes.
                     2. Escolha um conceito central que possa ser avaliado com uma pergunta objetiva.
                     3. Elabore um enunciado claro, contextualizado, com pelo menos 50 palavras, baseado nesse conceito.
                     4. Verifique se a **resposta correta não está explícita no enunciado**.
                     5. Crie 5 alternativas (a–e), sendo apenas uma correta.
                     6. Indique qual alternativa é a correta.
                     7. Justifique de forma detalhada por que a alternativa correta está certa.
                     8. Justifique de forma detalhada por que as demais alternativas estão incorretas.
                     9. Apresente quebras de linha entre as alternativas e entre as justificativas para ficar bem separado. 
                 
                     Evite copiar diretamente o conteúdo original. Não mencione nomes de documentos ou fontes no enunciado.
                     
                     {dificuldade}
                     
                     Conteúdo de base:
                     {input}

                     Apresente a questão no seguinte formato (cada alternativa deve estar em uma linha separada):
                     inserir uma linha horizontal
                     **Questão X**
                     
                     **Enunciado:** ...
                     
                     Sobre o exposto acima assinale a alternativa CORRETA: 
                     
                     a) [Elabore a primeira alternativa e insira uma quebra de linha.]
                     
                     b) [Elabore a segunda alternativa e insira uma quebra de linha.]
                     
                     c) [Elabore a terceira alternativa e insira uma quebra de linha.]
                     
                     d) [Elabore a quarta alternativa e insira uma quebra de linha.]
                     
                     e) [Elabore a quinta alternativa e insira uma quebra de linha.]
                     
                     **Alternativa correta:** ...
                     
                     **Justificativa (raciocínio passo a passo):**
                     
                     Confirme se realmente não é possível responder a questão com as informações presentes no enunciado
                        se sim, gere outra questão.
                     """
        )

    elif tipo_questao == "B":
        return PromptTemplate.from_template(
             """
                        Você é um professor de Ensino Superior e precisa elaborar exatamente {quantidade} questão(ões) 
                        de múltipla escolha com afirmativas.
                        
                        Siga o seguinte raciocínio passo a passo:
                        
                        1. Analise o conteúdo fornecido e identifique 4 ideias ou conceitos distintos que podem ser transformados em afirmativas.
                        2. Elabore um enunciado claro e contextualizado com pelo menos 50 palavras.
                        3. Escreva 4 afirmativas numeradas de I a IV.
                        4. Elabore 5 alternativas (a–e), com diferentes combinações de afirmativas corretas.
                        5. Verifique se **as afirmativas não indicam claramente qual alternativa é correta apenas pela redação**.
                        6. Indique qual alternativa é a correta.
                        7. Justifique de forma detalhada por que a alternativa correta está certa.
                        8. Justifique de forma detalhada por que as demais alternativas estão incorretas.
                        9. Apresente quebras de linha entre as alternativas e entre as justificativas para ficar bem separado. 
                                               
                        Evite copiar diretamente o conteúdo original. Não mencione o nome do documento.
                        
                        {dificuldade}
                        
                        Conteúdo de base:
                        {input}
                        
                        Apresente a questão no seguinte formato (cada alternativa deve estar em uma linha separada):
                        inserir uma linha horizontal
                        **Questão X**
                        **Enunciado:** ...
                        
                        I. [Elabore a primeira afirmativa e insira uma quebra de linha.]
                        
                        II. [Elabore a segunda afirmativa e insira uma quebra de linha.]
                        
                        III. [Elabore a terceira afirmativa e insira uma quebra de linha.]
                        
                        IV. [Elabore a quarta afirmativa e insira uma quebra de linha.]
                        
                        Assinale a alternativa que apresenta todas as afirmativas corretas.
                        
                        a) [Elabore a primeira alternativa e insira uma quebra de linha.]
                        
                        b) [Elabore a segunda alternativa e insira uma quebra de linha.]
                        
                        c) [Elabore a terceira alternativa e insira uma quebra de linha.]
                        
                        d) [Elabore a quarta alternativa e insira uma quebra de linha.]
                        
                        e) [Elabore a quinta alternativa e insira uma quebra de linha.]                        
                        
                        **Alternativa correta:** ...
                        
                        **Justificativa (raciocínio passo a passo):**
                        
                        Confirme se realmente não é possível responder a questão com as informações presentes no enunciado
                        se sim, gere outra questão.
                        """
        )

    elif tipo_questao == "C":
        return PromptTemplate.from_template(
            """
                                  Você é um professor de Ensino Superior e precisa elaborar {quantidade} questão(ões) 
                                  do tipo múltipla escolha com estrutura de asserção e razão.

                                  Siga o seguinte raciocínio passo a passo:

                                  1. Analise o conteúdo fornecido e selecione um conceito principal.
                                  2. Elabore um enunciado introdutório (contexto) com pelo menos 50 palavras.
                                  3. Crie uma **asserção (I)** e uma **razão (II)** relacionadas ao conceito.
                                  4. Elabore 5 alternativas (a–e) sobre a veracidade das asserções e se a razão justifica a asserção.
                                  5. Verifique se **o enunciado não antecipa a resposta correta nem contém pistas óbvias**.
                                  6. Indique qual alternativa é a correta.
                                  7. Justifique de forma detalhada por que a alternativa correta está certa.
                                  8. Justifique de forma detalhada por que as demais alternativas estão incorretas.
                                  9. Apresente quebras de linha entre as alternativas e entre as justificativas para ficar bem separado. 

                                  Evite copiar diretamente o conteúdo original. Não mencione o nome do documento.

                                  {dificuldade}
                                  Conteúdo de base:
                                  {input}

                                  Apresente a questão no seguinte formato (cada alternativa deve estar em uma linha separada):
                                  inserir uma linha horizontal
                                  **Questão X**
                                  **Enunciado:** ...

                                  Considerando as informações, avalie as asserções a seguir e a relação proposta entre elas.

                                  I. ... 

                                  PORQUE  

                                  II. ... 

                                  A respeito dessas asserções, assinale a opção CORRETA.

                                  a) As asserções I e II são proposições verdadeiras, e a II é uma justificativa correta da I.

                                  b) As asserções I e II são proposições verdadeiras, mas a II não é uma justificativa correta da I.

                                  c) A asserção I é uma proposição verdadeira, e a II é uma proposição falsa.

                                  d) A asserção I é uma proposição falsa, e a II é uma proposição verdadeira.

                                  e) As asserções I e II são proposições falsas.

                                  **Alternativa correta:** ...

                                  **Justificativa (raciocínio passo a passo):**

                                  Confirme se realmente não é possível responder a questão com as informações presentes no enunciado
                                  se sim, gere outra questão.
            """
        )

    else:
        raise ValueError("Tipo de questão inválido")
