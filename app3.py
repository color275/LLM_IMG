# export AWS_DEFAULT_REGION='us-west-2'
# nohup streamlit run app.py --server.port 8502 &

import streamlit as st
import fitz
import logging
import boto3
import json
import os
import re
import pymysql
import pandas as pd
from PIL import Image
import base64
import io
from dotenv import load_dotenv
from datetime import datetime
from langchain_community.chat_models import BedrockChat
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms.bedrock import Bedrock
from opensearchpy import OpenSearch, RequestsHttpConnection
from opensearchpy.helpers import bulk
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-west-2')

INIT_MESSAGE = {"role": "assistant",
                "type": "text",
                "content": """
안녕하세요. 저는 <font color='red'><b>Amazon Bedrock과 Claude3</b></font>를 활용해서 여러분들이 찾고 싶은 데이터를 대신 찾아줄 <i><b>[데이터가 궁금해]<i><b> 입니다. 
<br>아래와 같이 질문해보세요.
- <font color='#32CD32;'><b>어제 판매된 상품 기준으로 주문 금액 TOP 5 를 알려줘</b></font><br>
- <font color='#32CD32;'><b>지난 일주일간 주문 실적을 일 별로 알려줘</b></font><br>
- <font color='#32CD32;'><b>최근 5분 동안 총주문금액과 총주문수량을 분 단위로 알려줘</b></font><br>
- <font color='#32CD32;'><b>오늘 총 주문금액이 가장 적은 상품을 알려줘</b></font><br>
---
무엇을 도와드릴까요?"""}



################################################################################

load_dotenv()
opensearch_username = os.getenv('OPENSEARCH_USERNAME')
opensearch_password = os.getenv('OPENSEARCH_PASSWORD')
opensearch_endpoint = os.getenv('OPENSEARCH_ENDPOINT')
index_name = os.getenv('OPENSEARCH_INDEX_NAME')

bedrock_region = 'us-west-2'
stop_record_count = 100
record_stop_yn = False
bedrock_model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
bedrock_embedding_model_id = "amazon.titan-embed-text-v1"
################################################################################


def get_opensearch_cluster_client():
    opensearch_client = OpenSearch(
        hosts=[{
            'host': opensearch_endpoint,
            'port': 443
        }],
        http_auth=(opensearch_username, opensearch_password),
        index_name=index_name,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=30
    )
    return opensearch_client


def get_bedrock_client():
    bedrock_client = boto3.client(
        "bedrock-runtime", region_name=bedrock_region)
    return bedrock_client


def create_langchain_vector_embedding_using_bedrock(bedrock_client):
    bedrock_embeddings_client = BedrockEmbeddings(
        client=bedrock_client,
        model_id=bedrock_embedding_model_id)
    return bedrock_embeddings_client


def create_opensearch_vector_search_client(bedrock_embeddings_client, _is_aoss=False):
    docsearch = OpenSearchVectorSearch(
        index_name=index_name,
        embedding_function=bedrock_embeddings_client,
        opensearch_url=f"https://{opensearch_endpoint}",
        http_auth=(opensearch_username, opensearch_password),
        is_aoss=_is_aoss
    )
    return docsearch


def create_bedrock_llm():
    bedrock_llm = BedrockChat(
        model_id=bedrock_model_id, 
        model_kwargs={'temperature': 0},
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()]
        )
    return bedrock_llm


def get_bedrock_client():
    bedrock_client = boto3.client(
        "bedrock-runtime", region_name=bedrock_region)
    return bedrock_client


def create_vector_embedding_with_bedrock(text, bedrock_client):
    payload = {"inputText": f"{text}"}
    body = json.dumps(payload)
    modelId = "amazon.titan-embed-text-v1"
    accept = "application/json"
    contentType = "application/json"

    response = bedrock_client.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    embedding = response_body.get("embedding")
    return {"_index": index_name, "text": text, "vector_field": embedding}


def extract_sentences_from_pdf(opensearch_client, pdf_file, progress_bar, progress_text):
    try:
        logging.info(
            f"Checking if index {index_name} exists in OpenSearch cluster")

        exists = opensearch_client.indices.exists(index=index_name)

        if not exists:
            body = {
                'settings': {
                    'index': {
                        'number_of_shards': 3,
                        'number_of_replicas': 2,
                        "knn": True,
                        "knn.space_type": "cosinesimil"
                    }
                }
            }
            success = opensearch_client.indices.create(index_name, body=body)
            if success:
                body = {
                    "properties": {
                        "vector_field": {
                            "type": "knn_vector",
                            "dimension": 1536
                        },
                        "text": {
                            "type": "keyword"
                        }
                    }
                }
                success = opensearch_client.indices.put_mapping(
                    index=index_name,
                    body=body
                )

        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        all_records = []
        for page in doc:
            all_records.append(page.get_text())

        logging.info(f"PDF LIST 개수 : {len(all_records)}")

        total_records = len(all_records)
        processed_records = 0

        bedrock_client = get_bedrock_client()

        all_json_records = []

        for record in all_records:
            if record_stop_yn and processed_records > stop_record_count:

                success, failed = bulk(opensearch_client, all_json_records)
                break

            records_with_embedding = create_vector_embedding_with_bedrock(
                record, bedrock_client)
            all_json_records.append(records_with_embedding)

            processed_records += 1
            progress = int((processed_records / total_records) * 100)
            progress_bar.progress(progress)

            if processed_records % 500 == 0 or processed_records == len(all_records):

                success, failed = bulk(opensearch_client, all_json_records)
                all_json_records = []

        progress_text.text("완료")
        logging.info("임베딩을 사용하여 레코드 생성 완료")

        return total_records
    except Exception as e:
        print(str(e))
        st.error('PDF를 임베딩 하는 과정에서 오류가 발생되었습니다.')
        return 0

def scan_using_bedrock(image) :
    base64_encoded_image = get_image_base64(image)
    
    # 다음 [질문], [리뷰예시]를 참고하여 사진에 대한 설명과 느낀 점을 포함해 주세요.
    # prompt = """
    # 당신은 여행 중에 찍은 사진을 가지고 있습니다. 이 사진에 담긴 순간을 기반으로 여행 경험에 대한 리뷰를 작성해주세요.
    # 리뷰는 1인칭 시점에서 작성해주세요.
    # markdown 언어로 작성을 해주고, 강조할 부분이 있다면 markdown tag를 사용해주세요.
    # 200글자 내로 작성해주세요.
    
    # 아래와 같은 포맷으로 작성해주세요.
    # ### {타이틀}
    # {내용}
    # """

    prompt = """
    "여행 중 찍은 사진을 기반으로, 1인칭 시점에서 여행 경험 리뷰를 작성해주세요. 
    리뷰는 200글자 이내로 요약하며, Markdown 언어를 사용하여 포맷해주세요. 
    중요한 부분은 bold 태그로 강조하세요.
    한글로 작성해주세요.

    ## 여행 제목
    여기에 리뷰 내용을 써주세요. 내용은 간결하면서도 여행의 핵심 경험을 담아야 합니다."
    """

    # prompt = """
    # 상세하게 묘사된 이 이미지는 엑셀 형식으로 구성되어 있으며, 여러 행과 열에 걸쳐 다양한 숫자와 계산식이 포함되어 있습니다.
    # 각 셀에 적힌 숫자는 경제적 분석, 학술 연구 데이터, 또는 일상적인 가계부 계산과 같이 다양한 맥락에서 해석될 수 있습니다.
    # 이미지 분석을 통해 각 계산식의 적용 맥락과 가능한 의미를 설명하고, 이를 통해 얻을 수 있는 인사이트를 다양하게 제공해주세요.
    # markdown 언어로 작성을 해주고, 강조할 부분이 있다면 markdown tag를 사용해주세요.
    # 한글로 설명해주세요.
    # """

    payload = {
        "modelId": "anthropic.claude-3-sonnet-20240229-v1:0",
        "contentType": "application/json",
        "accept": "application/json",
        "body": {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 40960,
            "top_k": 250,
            "top_p": 0.999,
            "temperature": 1,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64_encoded_image
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        }
    }

    # Convert the payload to bytes
    body_bytes = json.dumps(payload['body']).encode('utf-8')

    # Invoke the model
    response = bedrock_runtime.invoke_model(
        body=body_bytes,
        contentType=payload['contentType'],
        accept=payload['accept'],
        modelId=payload['modelId']
    )

    # Process the response
    response_body = json.loads(response['body'].read())
    result = response_body['content'][0]['text']
    return result

# def resize_image(image, max_size=1048576, quality=90):
def resize_image(image, target_size_mb=1, quality=85):
    """
    이미지를 주어진 타겟 사이즈(메가바이트) 미만으로 리사이즈합니다.
    JPEG 포맷으로 압축하여 사이즈를 줄입니다.
    """
    # 타겟 사이즈를 바이트로 변환 (1MB = 1 * 1024 * 1024 바이트)
    target_size_bytes = target_size_mb * 1024 * 1024
    
    img_buffer = io.BytesIO()
    image.save(img_buffer, format='JPEG', quality=quality)
    img_size = img_buffer.tell()

    # 이미지 사이즈가 타겟보다 큰 경우 리사이즈
    while img_size > target_size_bytes:
        img_buffer = io.BytesIO()
        width, height = image.size
        # 이미지 크기를 10%씩 줄임
        image = image.resize((int(width * 0.9), int(height * 0.9)), Image.Resampling.LANCZOS)
        # 다시 저장하여 사이즈 체크
        image.save(img_buffer, format='JPEG', quality=quality)
        img_size = img_buffer.tell()

    # 버퍼의 포지션을 0으로 리셋
    img_buffer.seek(0)
    # BytesIO 객체를 PIL 이미지로 다시 변환
    return Image.open(img_buffer)

def get_image_base64(image, quality=85):
    """
    이미지 파일을 받아서 JPEG 포맷으로 압축하고,
    Base64 인코딩된 문자열로 변환합니다.
    `quality` 파라미터로 이미지의 압축 품질을 조절할 수 있습니다.
    """
    buffered = io.BytesIO()
    # JPEG 포맷으로 이미지 저장 및 품질 조절
    image.save(buffered, format="JPEG", quality=quality)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


def main():

    opensearch_client = get_opensearch_cluster_client()
    st.set_page_config(page_title='🤖 Chat with Bedrock', layout='wide')
    st.header(':blue[Review] _궁금해_ :sunglasses:', divider='rainbow')    


    with st.sidebar:
        st.sidebar.markdown(
            ':smile: **Createby:** chiholee@amazon.com', unsafe_allow_html=True)
        st.sidebar.markdown('---')
        st.title("IMG Upload")        
        img_file = st.file_uploader(
            "이미지를 업로드하세요.", type=['jpg', 'png', 'jpeg'])
        
        if img_file is not None :
            st.session_state['img_file'] = img_file
        
        st.sidebar.markdown('---')
        st.title("RAG Embedding")        
        pdf_file = st.file_uploader(
            "PDF 업로드를 통해 추가 학습을 할 수 있습니다.", type=["pdf"], key=None)

        if 'last_uploaded' not in st.session_state:
            st.session_state.last_uploaded = None

        if pdf_file is not None and pdf_file != st.session_state.last_uploaded:
            progress_text = st.empty()
            st.session_state['progress_bar'] = st.progress(0)
            progress_text.text("RAG(OpenSearch) 임베딩 중...")
            record_cnt = extract_sentences_from_pdf(
                opensearch_client, pdf_file, st.session_state['progress_bar'], progress_text)
            if record_cnt > 0:
                st.session_state['processed'] = True
                st.session_state['record_cnt'] = record_cnt
                st.session_state['progress_bar'].progress(100)
                st.session_state.last_uploaded = pdf_file
                st.success(f"{record_cnt} Vector 임베딩 완료!")

    if 'img_file' in st.session_state :
        col1, col2 = st.columns(2)

        image = Image.open(st.session_state['img_file'])        
        # image = resize_image(image)

        with col1:
            st.header("이미지")
            st.image(image, caption='Uploaded Image.', use_column_width=True)

        # 텍스트 입력 박스를 오른쪽 컬럼에 배치
        with col2:
            review = scan_using_bedrock(image)
            print(review)
            st.header("리뷰")
            # 사용자 입력을 위한 텍스트 에어리어
            # image_description = st.text_area("이미지에 대한 설명을 작성하세요.", height=300)
            st.markdown(review, unsafe_allow_html=True)



if __name__ == "__main__":
    main()
