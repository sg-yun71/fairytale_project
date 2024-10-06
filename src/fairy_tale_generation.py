import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

def chat_llm():
    """
    동화 생성에 사용되는 거대언어모델 생성 함수
    :return: 거대언어모델
    """
    load_dotenv('.env')

    # OpenAI API 호출을 통해 구동 시 사용
    llm = ChatOpenAI(
        model="gpt-4o",  # "gpt-4"로 변경 가능
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.7,  # 창의성 및 다양성 증가
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
    )

    return llm


def generate_story(llm):
    """
    키워드를 받아 새로운 동화를 생성하는 함수
    :param llm: 거대 언어 모델
    """
    # 사용자로부터 키워드 입력받기
    keywords = input("동화의 주제나 요소가 될 키워드를 입력하세요 (콤마로 구분): ")

    # 동화 생성 프롬프트 (키워드와 연관성을 강조)
    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                당신은 한국어로 동화를 만드는 AI입니다.
                주어진 키워드와 연관된 주제와 내용을 담은 재미있고 교훈적인 동화를 작성해주세요.
                동화에는 반드시 모든 키워드가 포함되어야 하며, 각 키워드와 관련된 이야기를 만들어주세요.
                동화는 반드시 한국어로 작성되어야 합니다.
                """,
            ),
            ("human", "키워드: {keywords}"),
        ]
    )

    # 프롬프트를 한국어로 생성
    prompt = prompt_template.format_prompt(keywords=keywords)

    # 동화 생성
    response = llm.invoke(prompt.to_messages())
    print("생성된 동화:\n")
    print(response.content)


def run():
    """
    동화 생성 시작 함수
    """
    # 채팅에 사용할 거대언어모델(LLM) 선택
    llm = chat_llm()

    # 동화 생성
    generate_story(llm)


if __name__ == "__main__":
    run()