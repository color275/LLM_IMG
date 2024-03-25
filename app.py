# export AWS_DEFAULT_REGION='us-west-2'
# nohup streamlit run app.py args --server.fileWatcherType none --server.port 8504 &

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

################################################################################

load_dotenv()
opensearch_username = os.getenv('OPENSEARCH_USERNAME')
opensearch_password = os.getenv('OPENSEARCH_PASSWORD')
opensearch_endpoint = os.getenv('OPENSEARCH_ENDPOINT')
index_name = os.getenv('OPENSEARCH_INDEX_NAME')
mysql_host = os.getenv('MYSQL_HOST')
mysql_port = os.getenv('MYSQL_PORT')
mysql_user = os.getenv('MYSQL_USER')
mysql_password = os.getenv('MYSQL_PASSWORD')
mysql_db = os.getenv('MYSQL_DB')

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
    # claude-2 이하
    # bedrock_llm = Bedrock(
    #     model_id=model_version_id,
    #     client=bedrock_client,
    #     model_kwargs={'temperature': 0}
    #     )
    # bedrock_llm = BedrockChat(model_id=model_version_id, model_kwargs={'temperature': 0}, streaming=True)

    bedrock_llm = BedrockChat(
        model_id=bedrock_model_id, 
        model_kwargs={'temperature': 1,
                      "top_k": 250,
                      "top_p": 0.999,
                      "max_tokens": 4096
                      },
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

        # URL Scraping
        # doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        # all_text = ""
        # for page in doc:
        #     all_text += page.get_text()
        # doc.close()
        # all_records = re.split(r'(?<=[.!?])\s+', all_text)

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


def find_answer_in_sentences(image_description, user_keyword):
    try:
        question = f"""
        이 이미지는 "{image_description}"와 관련이 있습니다. 
        사용자가 제시한 사용자 키워드는 "{user_keyword}" 입니다.
        여행제목의 글자색은 흰색으로 표시해줘.
        리뷰내용에 사용자 키워드와 비슷한 문구는 markdown의 태그를 활용해서 녹색으로 표시해줘.
        만약 제공된 정보 중 사용자 키워드와 관계된 내용이 없다면 제공된 정보를 사용하지 말고 키워드 중심으로 적절하게 리뷰를 400자 내로 작성해주세요.
        다른 안내 문장은 제공하지 말고 여행 리뷰만 작성해줘."""

        # question = f"""
        # 당신은 여행 중 찍은 사진으로 "{image_description}" 라고 리뷰내용을 이미 작성했다.
        # 추가 제공된 정보에 리뷰내용을 보충할 정보가 있다면 보충해줘.
        # 보충할 내용이 없으면 리뷰내용을 그대로 출력해줘.
        # 리뷰내용에 사용자 키워드인 "{user_keyword} 와 비슷한 문구는 markdown의 태그를 활용해서 녹색으로 표시해줘
        # 여행제목의 글자색은 흰색으로 표시해줘.
        # 여행리뷰만 보여줘.
        # 앞 뒤로 안내 멘트는 없애줘.
        # """

        print("## question : ",question)

        bedrock_client = get_bedrock_client()
        bedrock_llm = create_bedrock_llm()

        bedrock_embeddings_client = create_langchain_vector_embedding_using_bedrock(
            bedrock_client)

        opensearch_vector_search_client = create_opensearch_vector_search_client(
            bedrock_embeddings_client)
        
        
        # Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. don't include harmful content        
        # SQL 생성 시 최대 10개의 레코드만 표시되도록 SQL의 가장 하단에 Limit 10을 SQL에 포함시켜줘. 그리고 Limit 10을 임의로 추가했다고 초록색 글자로 안내해줘.
        # 제공된 정보 중 사용자 키워드와 관련있는 정보만 사용해줘.
        # 다른 안내 문장은 제공하지 말고 여행 리뷰만 작성해줘.

        prompt_template = """
        Use the following pieces of context to answer the question at the end.       

        {context}
        여행 제목의 글자 크기는 markdown의 H3 사이즈를 사용해줘.

        Question: {question}
        Answer:
        ## 여행제목
        여기에 리뷰내용을 써주세요. 내용은 간결하면서도 여행의 핵심 경험을 담아야 합니다."""
        
        prompt_template = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        # prompt = prompt_template.format(
        #     context=CONTEXT_DATA, question=question)
        
        # print("# prompt : ", prompt)


        logging.info(
            f"Starting the chain with KNN similarity using OpenSearch, Bedrock FM {bedrock_model_id}, and Bedrock embeddings with {bedrock_embedding_model_id}")

        qa = RetrievalQA.from_chain_type(llm=bedrock_llm,
                                         chain_type="stuff",
                                         retriever=opensearch_vector_search_client.as_retriever(),
                                         return_source_documents=True,
                                         chain_type_kwargs={
                                             "prompt": prompt_template, "verbose": True},
                                         verbose=True)

        # response = qa({"context": CONTEXT_DATA, "query": question},
        #               return_only_outputs=False)
        response = qa(question,
                      return_only_outputs=False)

        source_documents = response.get('source_documents')
        # logging.info(f"The answer from Bedrock {bedrock_model_id} is: {response.get('result')}")
        return f"{response.get('result')}"
    except Exception as e:
        if 'index_not_found_exception' in str(e):
            st.error('인덱스를 찾을 수 없습니다. PDF 파일을 업로드 했는지 확인해주세요')
        else:
            print(str(e))
            st.error('답변을 찾는 과정에서 예상치 못한 오류가 발생했습니다.')
        return "오류로 인해 답변을 제공할 수 없습니다."

def connect_to_database():
    return pymysql.connect(
        host=mysql_host,
        port=int(mysql_port),
        user=mysql_user,
        password=mysql_password,
        database=mysql_db,
        charset='utf8mb4'
    )

# SQL 쿼리 실행 및 결과를 데이터프레임으로 변환


def execute_query_and_return_df(sql):
    conn = connect_to_database()
    try:
        with conn.cursor() as cursor:
            cursor.execute(sql)
            result = cursor.fetchall()
            df = pd.DataFrame(result, columns=[i[0]
                              for i in cursor.description])
    finally:
        conn.close()
    return df

def scan_using_bedrock(image, user_keyword) :
    base64_encoded_image = get_image_base64(image)
    
    # 리뷰는 200글자 이내로 요약하며, Markdown 언어를 사용하여 포맷해주세요. 
    # 중요한 부분은 bold 태그로 강조하세요.
    # ## 여행 제목
    # 여기에 리뷰 내용을 써주세요. 내용은 간결하면서도 여행의 핵심 경험을 담아야 합니다."
    prompt = f"""
    "여행 중 찍은 사진을 기반으로, 1인칭 시점에서 여행 경험 리뷰를 작성해주세요. 
    이 사진의 키워드는 "{user_keyword}" 이며 키워드를 중심으로 리뷰를 작성해주세요.
    리뷰는 200글자 이내로 요약해주세요.
    한글로 작성해주세요.
    """
    # prompt = f"""
    # 당신은 여행 중에 기억에 남기고 싶은 순간을 사진으로 찍었습니다.
    # 그리고 사진에 대해 "{user_keyword}" 주제로 리뷰를 작성합니다.
    # 리뷰는 200글자 이내로 요약해주세요.
    # 1인칭 시점으로 작성해주세요.
    # 한글로 작성해주세요.
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

    # 기존 업로드 문서 삭제
    # if st.sidebar.button("기존 업로드 문서 삭제"):
    #     response = opensearch_client.delete_opensearch_index(opensearch_client, index_name)
    #     # st.session_state['question'] = ""  # 질문 세션 상태 초기화
    #     if response:
    #         logging.info("OpenSearch index successfully deleted")
    #         st.sidebar.success("OpenSearch 인덱스가 성공적으로 삭제되었습니다.")  # 성공 알림 추가

    opensearch_client = get_opensearch_cluster_client()
    
    # 기존 업로드 문서 삭제
    # opensearch_client.indices.delete(index=index_name)
    
    st.set_page_config(page_title='🤖 Chat with Bedrock', layout='wide')
    # st.header('_Chatbot_ using :blue[OpenSearch] :sunglasses:', divider='rainbow')
    st.header(':blue[리뷰가] _궁금해_ :sunglasses:', divider='rainbow')    

    if 'qa_history' not in st.session_state:
        st.session_state.qa_history = []

    with st.sidebar:
        st.sidebar.markdown(
            ':smile: **Createby:** chiholee@amazon.com', unsafe_allow_html=True)
        
        st.sidebar.markdown('---')
        user_keyword = st.text_input('(필수)여행 키워드를 입력하세요')
        
        if user_keyword is not None :
            st.session_state['user_keyword'] = user_keyword

        st.title("IMG Upload")        
        img_file = st.file_uploader(
            "이미지를 업로드하세요.", type=['jpg', 'png', 'jpeg'])
        
        if img_file is not None :
            st.session_state['img_file'] = img_file

        st.sidebar.markdown('---')
        st.title("RAG Embedding")
        st.caption("관리자 only")
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

    
    try :
        if 'img_file' in st.session_state and not user_keyword :
            st.error('파일을 업로드하기 전에 키워드를 입력해주세요.')            

        elif 'img_file' in st.session_state :
            
            with st.spinner('리뷰 작성중...:hourglass:'):
                col1, col2 = st.columns(2)

                image = Image.open(st.session_state['img_file'])        
                # image = resize_image(image)

                image_description = scan_using_bedrock(image, st.session_state['user_keyword'])
                review = find_answer_in_sentences(image_description, st.session_state['user_keyword'])

                with col1:
                    st.subheader("📷 Image")
                    st.image(image, caption='Uploaded Image.', use_column_width=True)

                with col2:
                    st.subheader("🔍 Review")
                    st.markdown(review, unsafe_allow_html=True)

                
                st.markdown('---')
                st.subheader(":thinking_face: 리뷰 생성 과정")
                ref1, ref2, ref3 = st.columns([1,3,3])
                with ref1 :
                    st.write("📝 입력 Keyword")
                    st.markdown(st.session_state['user_keyword'], unsafe_allow_html=True)
                with ref2 :
                    st.write("📝 Keyword + Image 기반 LLM")
                    st.markdown(image_description, unsafe_allow_html=True)
                with ref3 :
                    st.write("📝 Keyword + Image + RAG 기반 LLM")
                    st.markdown(review, unsafe_allow_html=True)
    except Exception as e :
        st.error('죄송합니다. 리뷰 작성 중 오류가 발생했습니다. 다시 한번 실행해주세요.', icon="🚨")
    
    

if __name__ == "__main__":
    main()
