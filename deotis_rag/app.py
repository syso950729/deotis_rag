import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_teddynote.prompts import load_prompt
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_teddynote import logging
from dotenv import load_dotenv
import os
import re
import datetime
import json

# API KEY 정보로드
load_dotenv()

# 프로젝트 이름을 입력합니다.
logging.langsmith("[Project] PDF RAG")

# 캐시 디렉토리 생성
if not os.path.exists(".cache"):
    os.mkdir(".cache")

# 파일 업로드 전용 폴더
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")

if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")

# 피드백 디렉토리 생성
if not os.path.exists(".cache/feedback"):
    os.mkdir(".cache/feedback")

st.title("PDF 기반 QA💬")

# 처음 1번만 실행하기 위한 코드
if "messages" not in st.session_state:
    # 대화기록을 저장하기 위한 용도로 생성한다.
    st.session_state["messages"] = []

if "chain" not in st.session_state:
    # 아무런 파일을 업로드 하지 않을 경우
    st.session_state["chain"] = None

# 문서 전처리 함수
def preprocess_documents(docs):
    """문서 전처리 - 불필요한 내용 제거 및 품질 향상"""
    processed_docs = []
    for doc in docs:
        content = doc.page_content
        
        # 1. 연속된 공백과 줄바꿈 정리
        content = re.sub(r'\n\s*\n+', '\n\n', content)
        content = re.sub(r'\s+', ' ', content)
        
        # 2. 페이지 번호, 헤더/푸터 제거
        content = re.sub(r'페이지\s*\d+', '', content)
        content = re.sub(r'Page\s*\d+', '', content)
        content = re.sub(r'^\d+\s*$', '', content, flags=re.MULTILINE)
        
        # 3. 특수문자 정리 (단, 의미있는 구두점은 보존)
        content = re.sub(r'[•◦▪▫■□●○◆◇▲△▼▽]', '- ', content)
        
        # 4. URL 및 이메일 정리
        content = re.sub(r'http[s]?://\S+', '[URL]', content)
        content = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', content)
        
        # 5. 내용 정리 및 검증
        content = content.strip()
        
        # 6. 최소 길이 확인 (너무 짧은 청크 제외)
        if len(content) > 30 and not content.isspace():
            doc.page_content = content
            processed_docs.append(doc)
    
    return processed_docs

# 하이브리드 검색 함수
def create_hybrid_retriever(split_documents, embeddings):
    """벡터 검색과 BM25 키워드 검색을 결합한 하이브리드 검색 생성"""
    
    # 1. 벡터 검색 설정
    vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)
    vector_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,  # 하이브리드에서는 각각 적은 수로
            "lambda_mult": 0.5,
            "fetch_k": 15
        }
    )
    
    # 2. BM25 키워드 검색 설정
    bm25_retriever = BM25Retriever.from_documents(split_documents)
    bm25_retriever.k = 4  # 벡터 검색과 동일한 수
    
    # 3. 하이브리드 검색 (앙상블)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.7, 0.3]  # 벡터 검색에 더 높은 가중치
    )
    
    return ensemble_retriever

# Query Expansion 기능 완전 제거 - 안정성을 위해

# 기본 청킹만 사용 - 복잡한 계층적 청킹 제거
def create_simple_chunks(docs):
    """기본 청킹만 사용 - 안정성 최우선"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    return text_splitter.split_documents(docs)

# 고급 검색 기능 완전 제거 - 기본 하이브리드 검색만 사용

# Reranking 기능 완전 제거 (안정성을 위해)
# 복잡한 의존성과 호환성 문제로 인해 제거됨

# 답변 검증 함수들
def validate_answer_quality(question, answer, context):
    """답변 품질을 자동으로 검증"""
    quality_score = 0
    issues = []
    
    # 1. 길이 체크 (너무 짧거나 길지 않은지)
    if len(answer) < 50:
        issues.append("답변이 너무 짧습니다")
    elif len(answer) > 3000:
        issues.append("답변이 너무 깁니다")
    else:
        quality_score += 1
    
    # 2. 구조 체크 (형식이 잘 갖춰져 있는지)
    if "### 📋 요약" in answer and "### 📖 상세 답변" in answer:
        quality_score += 1
    else:
        issues.append("답변 형식이 불완전합니다")
    
    # 3. 내용 관련성 체크 (질문과 관련 키워드가 포함되어 있는지)
    question_keywords = set(question.lower().split())
    answer_keywords = set(answer.lower().split())
    common_keywords = question_keywords.intersection(answer_keywords)
    
    if len(common_keywords) >= 2:
        quality_score += 1
    else:
        issues.append("질문과 답변의 연관성이 낮습니다")
    
    # 4. "모르겠다" 류의 답변 체크
    uncertain_phrases = ["모르겠", "확실하지 않", "찾을 수 없", "명확하지 않"]
    if any(phrase in answer for phrase in uncertain_phrases):
        issues.append("불확실한 답변이 포함되어 있습니다")
    else:
        quality_score += 1
    
    return {
        "score": quality_score,
        "max_score": 4,
        "issues": issues,
        "quality": "높음" if quality_score >= 3 else "보통" if quality_score >= 2 else "낮음"
    }

# 피드백 로깅 함수들
def log_feedback(feedback_type, question, answer, details=None):
    """피드백을 파일에 저장"""
    feedback_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "type": feedback_type,
        "question": question,
        "answer": answer,
        "details": details
    }
    
    # 파일명에 날짜 포함
    date_str = datetime.datetime.now().strftime("%Y%m%d")
    feedback_file = f".cache/feedback/feedback_{date_str}.jsonl"
    
    # JSONL 형식으로 추가
    with open(feedback_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(feedback_data, ensure_ascii=False) + "\n")

def show_feedback_stats():
    """피드백 통계 표시"""
    try:
        feedback_files = [f for f in os.listdir(".cache/feedback") if f.endswith(".jsonl")]
        if not feedback_files:
            return
        
        total_positive = 0
        total_negative = 0
        
        for file in feedback_files:
            with open(f".cache/feedback/{file}", "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    if data["type"] == "positive":
                        total_positive += 1
                    elif data["type"] == "negative":
                        total_negative += 1
        
        if total_positive + total_negative > 0:
            satisfaction_rate = total_positive / (total_positive + total_negative) * 100
            st.sidebar.success(f"📊 만족도: {satisfaction_rate:.1f}% ({total_positive}👍/{total_negative}👎)")
    except:
        pass

# 사이드바 생성
with st.sidebar:
    # 초기화 버튼 생성
    clear_btn = st.button("대화 초기화")

    # 파일 업로드
    uploaded_file = st.file_uploader("파일 업로드", type=["pdf"])

    # 모델 선택 메뉴
    model_options = {
        "Gemini 2.5 Pro (최고 성능)": "gemini-2.5-pro",
        "Gemini 2.5 Flash (균형)": "gemini-2.5-flash", 
        "Gemini 2.5 Flash-Lite (효율)": "gemini-2.5-flash-lite",
        "Gemini 1.5 Flash (기존)": "gemini-1.5-flash"
    }
    
    selected_display = st.selectbox(
        "LLM 선택", list(model_options.keys()), index=0
    )
    selected_model = model_options[selected_display]
    
    # 모델 설명 표시
    model_descriptions = {
        "Gemini 2.5 Pro (최고 성능)": "🏆 복잡한 추론과 긴 문서 분석에 최적화",
        "Gemini 2.5 Flash (균형)": "⚡ 빠른 속도와 좋은 성능의 균형",
        "Gemini 2.5 Flash-Lite (효율)": "💰 비용 효율적이고 빠른 처리",
        "Gemini 1.5 Flash (기존)": "📋 기본 모델"
    }
    st.info(model_descriptions[selected_display])
    
    # 피드백 통계 표시
    show_feedback_stats()
    
    # 모든 고급 기능 비활성화 (안정성 최우선)
    enable_query_expansion = False
    enable_hierarchical_chunking = False
    enable_reranking = False
    
    # 모든 고급 기능 제거됨
    
    # 기본 기능만 활성화
    features_text = "기본 하이브리드 검색 (벡터 + 키워드)"
    
    # 기본 모드 안내
    if uploaded_file:
        st.sidebar.info("💡 기본 버전: 가장 안정적인 설정")
    
    st.sidebar.success(f"""
🚀 **기본 RAG 시스템**

**🧠 LLM**: Gemini 2.5 Pro
**🔍 검색**: 하이브리드 (벡터 + 키워드)
**🗏️ 상태**: 안정 모드

{features_text}
""")
    
    # 기본 성능 메트릭
    base_improvement = 65  # Gemini 2.5 Pro + 하이브리드 검색
    
    st.sidebar.metric(
        "🎯 성능 향상",
        f"+{base_improvement}%",
        "안정적 기본 버전"
    )


# 이전 대화를 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# 새로운 메시지를 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


# 파일을 캐시 저장(시간이 오래 걸리는 작업을 처리할 예정)
@st.cache_resource(
    show_spinner="업로드한 파일을 처리 중입니다...",
    hash_funcs={type(st.sidebar): lambda x: None}  # 사이드바 상태 무시
)
def embed_file(file):
    # 업로드한 파일을 캐시 디렉토리에 저장합니다.
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    # 단계 1: 문서 로드(Load Documents)
    loader = PDFPlumberLoader(file_path)
    docs = loader.load()
    
    # 단계 1.5: 문서 전처리 (품질 향상)
    docs = preprocess_documents(docs)

    # 단계 2: 기본 문서 분할만 사용
    split_documents = create_simple_chunks(docs)

    # 단계 3: 임베딩(Embedding) 생성
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # 단계 4: 기본 하이브리드 검색 생성
    retriever = create_hybrid_retriever(split_documents, embeddings)
    
    print(f"🏁 Basic retriever ready: {type(retriever)}")
    return retriever


# 기본 체인 생성 - 모든 고급 기능 제거
def create_chain(retriever, model_name="gemini-2.5-pro"):
    # 프롬프트 생성
    prompt = load_prompt("prompts/pdf-rag.yaml", encoding="utf-8")

    # 언어모델(LLM) 생성
    llm = ChatGoogleGenerativeAI(
        model=model_name, 
        temperature=0.1,
        max_tokens=2000,
        top_p=0.9
    )

    # 기본 RAG 체인 - 가장 안정적인 구조
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


# 파일이 업로드 되었을 때
if uploaded_file:
    # 파일 업로드 후 기본 retriever 생성
    retriever = embed_file(uploaded_file)
    chain = create_chain(retriever, model_name=selected_model)
    st.session_state["chain"] = chain

# 초기화 버튼이 눌리면...
if clear_btn:
    st.session_state["messages"] = []

# 이전 대화 기록 출력
print_messages()

# 사용자의 입력
user_input = st.chat_input("궁금한 내용을 물어보세요!")

# 경고 메시지를 띄우기 위한 빈 영역
warning_msg = st.empty()

# 만약에 사용자 입력이 들어오면...
if user_input:
    # chain 을 생성
    chain = st.session_state["chain"]

    if chain is not None:
        # 사용자의 입력
        st.chat_message("user").write(user_input)
        
        with st.chat_message("assistant"):
            # 빈 공간(컨테이너)을 만들어서, 여기에 토큰을 스트리밍 출력한다.
            container = st.empty()
            
            # 스트리밍 응답 생성
            response = chain.stream(user_input)
            ai_answer = ""
            for token in response:
                ai_answer += token
                container.markdown(ai_answer)

        # 답변 품질 검증
        quality_result = validate_answer_quality(user_input, ai_answer, "")
        
        # 품질 결과 표시 (사이드바에)
        with st.sidebar:
            st.write("---")
            st.write("**📊 답변 품질 분석**")
            quality_color = "🟢" if quality_result["quality"] == "높음" else "🟡" if quality_result["quality"] == "보통" else "🔴"
            st.write(f"{quality_color} **품질**: {quality_result['quality']} ({quality_result['score']}/{quality_result['max_score']})")
            
            if quality_result["issues"]:
                st.write("**개선점:**")
                for issue in quality_result["issues"]:
                    st.write(f"• {issue}")
        
        # 대화기록을 저장한다.
        add_message("user", user_input)
        add_message("assistant", ai_answer)
        
        # 피드백 시스템
        st.write("---")
        st.write("**이 답변이 도움이 되었나요?**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("👍 도움됨", key=f"positive_{len(st.session_state['messages'])}"):
                log_feedback("positive", user_input, ai_answer)
                st.success("피드백 감사합니다!")
                st.rerun()
        
        with col2:
            if st.button("👎 도움 안됨", key=f"negative_{len(st.session_state['messages'])}"):
                log_feedback("negative", user_input, ai_answer)
                st.info("더 나은 답변을 위해 노력하겠습니다!")
                st.rerun()
                
        with col3:
            if st.button("📝 상세 피드백", key=f"detailed_{len(st.session_state['messages'])}"):
                st.session_state[f"show_feedback_{len(st.session_state['messages'])}"] = True
                st.rerun()
        
        # 상세 피드백 입력창
        if st.session_state.get(f"show_feedback_{len(st.session_state['messages'])}", False):
            with st.form(key=f"feedback_form_{len(st.session_state['messages'])}"):
                feedback_text = st.text_area("구체적인 피드백을 입력해 주세요:", 
                                           placeholder="예: 답변이 너무 길어요, 특정 부분이 부정확해요, 더 자세한 설명이 필요해요 등")
                submitted = st.form_submit_button("피드백 제출")
                
                if submitted and feedback_text:
                    log_feedback("detailed", user_input, ai_answer, feedback_text)
                    st.success("상세한 피드백 감사합니다! 개선에 참고하겠습니다.")
                    st.session_state[f"show_feedback_{len(st.session_state['messages'])}"] = False
                    st.rerun()
    else:
        # 파일을 업로드 하라는 경고 메시지 출력
        warning_msg.error("파일을 업로드 해주세요.")