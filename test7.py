import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import json

# 페이지 1에서 사용되는 함수
@st.cache_data
def load_model_and_data():
    # 데이터 및 모델 로드
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    df = pd.read_csv('wellness_dataset.csv')
    df['embedding'] = df['embedding'].apply(json.loads)
    return model, df

def generate_response(user_input, df, model):
    embedding = model.encode(user_input)
    df['distance'] = df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())
    answer = df.loc[df['distance'].idxmax()]
    return answer['챗봇']

def page1(model, df):
    st.title("인마고 :red[Q&A Chat] :robot_face:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    # 사이드바는 제거되었습니다.

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant", 
                                        "content": "안녕하세요! 인천전자마이스터고등학교에 대해 궁금하신 것이 있으면 언제든 물어봐주세요!"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 채팅 내용을 텍스트 파일로 저장
    if st.button('채팅 내용 저장'):
        with open('chat_history.txt', 'w', encoding='utf-8') as f:
            for content in st.session_state.chat_history:
                f.write(f"{content['role']}: {content['message']}\n")
        st.success('채팅 내용이 chat_history.txt 파일에 저장되었습니다.')

    # 챗봇 로직
    if query := st.chat_input("질문을 입력해주세요."):
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            response = generate_response(query, df, model)

            st.markdown(response)
            # 필요한 경우 여기에 소스 문서 정보를 포함할 수 있습니다.

        # 챗봇 메시지를 채팅 기록에 추가합니다.
        st.session_state.messages.append({"role": "assistant", "content": response})
        # 채팅 내용을 텍스트 파일에 추가합니다.
        st.session_state.chat_history.append({"role": "user", "message": query})
        st.session_state.chat_history.append({"role": "assistant", "message": response})

# 페이지 2에서 사용되는 함수
def page2():
    st.title("Page 2")
    st.write("Hello World!")

    # 버튼들의 동작 정의
    if st.button("Button 1"):
        st.image("D:\Files\MDP\MdpProject_0506\dog.jpg", caption="Dog Image", use_column_width=True)
    if st.button("Button 2"):
        st.image("D:\Files\MDP\MdpProject_0506\cat.jpg", caption="Cat Image", use_column_width=True)
    if st.button("Button 3"):
        st.image("D:\Files\MDP\MdpProject_0506\duck.jpg", caption="Duck Image", use_column_width=True)

# 메인 함수
def main():
    st.title("MDP YUBIN test7.py")

    # 메인 페이지 버튼 설정
    if st.button("ChatBot"):
        st.session_state.page = "page1"  # ChatBot 버튼 클릭 시 페이지 1로 이동
    if st.button("Map"):
        st.session_state.page = "page2"  # Map 버튼 클릭 시 페이지 2로 이동

    # 페이지 이동 로직
    if "page" not in st.session_state:
        st.session_state.page = "main"

    if st.session_state.page == "page1":
        model, df = load_model_and_data()
        page1(model, df)
        if st.button("Go back to Main Page"):
            st.session_state.page = "main"  # 페이지 1에서 메인 페이지로 돌아가는 버튼
    elif st.session_state.page == "page2":
        page2()
        if st.button("Go back to Main Page"):
            st.session_state.page = "main"  # 페이지 2에서 메인 페이지로 돌아가는 버튼

if __name__ == "__main__":
    main()
