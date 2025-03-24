__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import pysqlite3 as sqlite3
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import boto3

# 모델 로드
@st.cache_resource
def load_model():
    return SentenceTransformer("BAAI/bge-m3")

model = load_model()

# DB 로드
@st.cache_resource
def load_db():
    client_chroma = chromadb.PersistentClient(path="./database")
    return client_chroma.get_or_create_collection("database")

collection = load_db()

# 영상 검색 함수 (날짜 추가)
def find_videos(query, date=None, top_k=5):
    query_vector = model.encode(query)
    
    # 날짜 필터링 (파티셔닝)
    where_clause = {"date": date} if date else None
    
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    videos = []
    for doc, meta, dist in zip(results['documents'][0], results['metadatas'][0], results['distances'][0]):
        videos.append({
            "video_id": meta["video_id"],
            "distance": dist,
            "description": doc,
        })
    return videos

# Cloudflare R2 Pre-signed URL 생성 함수
def generate_presigned_url(video_id):
    import boto3

    r2_client = boto3.client(
        's3',
        endpoint_url=f'https://c5b8240dc0c186f421abddb091e2b049.r2.cloudflarestorage.com',
        aws_access_key_id=st.secrets["R2_ACCESS_KEY"],
        aws_secret_access_key=st.secrets["R2_SECRET_KEY"]
    )

    url = r2_client.generate_presigned_url(
        'get_object',
        Params={
            'Bucket': st.secrets["R2_BUCKET_NAME"],
            'Key': f"{video_id}.mp4"
        },
        ExpiresIn=600
    )
    return url

# Streamlit UI 구성
st.title("영상 검색 프로토타입 🔍")

query = st.text_input("검색할 영상 내용을 입력하세요.", "신호등과 교차로가 있는 영상")

if st.button("영상 찾기"):
    with st.spinner('검색 중입니다...'):
        results = find_videos(query)

    st.subheader("🔍 가장 유사한 영상 결과")
    for video in results:
        video_url = generate_presigned_url(f"{video['video_id']}")

        st.video(video_url)
        st.markdown(f"**영상 ID:** `{video['video_id']}`\n"
                    f"**유사도 점수:** `{video['distance']:.4f}`\n"
                    f"**설명:** {video['description']}\n---")
