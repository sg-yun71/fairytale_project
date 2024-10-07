import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings  # 수정된 import 경로
from langchain_community.vectorstores import Chroma  # 수정된 import 경로
import json
from langchain.schema import Document

# 환경 변수 로드
load_dotenv('.env')

def load_processed_data():
    """
    전처리된 JSON 데이터를 로드하는 함수
    :return: 로드된 데이터
    """
    with open('data/processed_stories.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def split_text(data):
    """
    텍스트 데이터를 청크로 분할하는 함수
    :param data: 전처리된 데이터
    :return: 청크로 분할된 텍스트
    """
    rc_text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        separators=["\n\n", "\n", " "],
        chunk_size=2000,
        chunk_overlap=500,
        encoding_name="o200k_base",
        model_name="gpt-4o"
    )
    
    # 데이터에서 "description" 키를 사용하여 텍스트 추출하고 Document 객체로 감싸기
    texts = [Document(page_content=story.get("description", "")) for story in data]
    text_documents = rc_text_splitter.split_documents(texts)
    return text_documents

def create_embedding_model():
    """
    임베딩 모델을 생성하는 함수
    :return: 생성된 임베딩 모델
    """
    model_name = "jhgan/ko-sroberta-multitask"  # 한국어 모델
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return model

def embed_documents(docs, model, save_directory="./chroma_db"):
    """
    분할된 문서를 임베딩하는 함수
    :param docs: 청크로 분할된 텍스트
    :param model: 임베딩 모델
    :param save_directory: 임베딩 저장 경로
    :return: Chroma 데이터베이스
    """
    import shutil

    # 벡터저장소가 이미 존재하면 삭제
    if os.path.exists(save_directory):
        shutil.rmtree(save_directory)
    
    db = Chroma.from_documents(docs, model, persist_directory=save_directory)
    db.persist()
    return db

def create_llm():
    """
    거대 언어 모델(LLM)을 생성하는 함수
    :return: 생성된 LLM
    """
    llm = ChatOpenAI(
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.7,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()]
    )
    return llm

def generate_story(llm, db, keywords):
    """
    키워드와 관련된 동화를 생성하는 함수
    :param llm: 거대 언어 모델
    :param db: Chroma 데이터베이스
    :param keywords: 동화 생성에 사용될 키워드
    :return: 생성된 동화 내용
    """
    # 임베딩된 데이터베이스에서 키워드와 연관된 내용을 검색
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={'k': 3, 'fetch_k': 5}
    )
    # 키워드를 바탕으로 가장 관련성이 높은 컨텍스트를 검색
    context_docs = retriever.get_relevant_documents(keywords)
    context_text = "\n".join(doc.page_content for doc in context_docs)

    # 프롬프트 생성 (키워드와 관련된 동화를 생성하도록 요청)
    prompt_text = f"""
    당신은 어린이를 위한 동화를 만드는 AI입니다.
    주어진 키워드를 모두 포함하여 재미있고 교훈적인 동화를 작성하세요.
    동화는 한국어로 작성되어야 하며 어린이가 이해할 수 있는 내용이어야 합니다.

    키워드: {keywords}
    참고 자료:
    {context_text}
    """

    # LLM을 사용하여 동화 생성
    response = llm.invoke([{"role": "system", "content": prompt_text}])
    print(f"\n생성된 동화:\n{response}")

    return response

    # LLM을 사용하여 동화 생성
    response = llm.invoke(prompt.format_prompt(context=context, keywords=keywords).to_messages())
    print(f"\n생성된 동화:\n{response}")

    return response

def run():
    """
    전체 프로그램 실행
    """
    # 전처리된 데이터 로드
    data = load_processed_data()

    # 데이터 분할
    chunks = split_text(data)

    # 임베딩 모델 생성
    embedding_model = create_embedding_model()

    # 데이터 임베딩 및 데이터베이스 생성
    db = embed_documents(chunks, embedding_model)

    # LLM 생성
    llm = create_llm()

    # 키워드 입력받기
    keywords = input("동화의 주제나 요소가 될 키워드를 입력하세요 (콤마로 구분): ")

    # 동화 생성
    generate_story(llm, db, keywords)

if __name__ == "__main__":
    run()
