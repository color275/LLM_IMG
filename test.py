import streamlit as st
import boto3
from PIL import Image
import base64
import json
import io
 
bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-west-2')

def scan_using_bedrock(image) :
    base64_encoded_image = get_image_base64(image)
    prompt = """
    이미지의 분위기를 한글로 작성해줘
    """

    payload = {
        "modelId": "anthropic.claude-3-sonnet-20240229-v1:0",
        "contentType": "application/json",
        "accept": "application/json",
        "body": {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4096,
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

def get_image_base64(image):
    """이미지 파일을 받아서 Base64 인코딩된 문자열로 변환"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")  # 혹은 업로드된 이미지의 원래 포맷을 사용
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

# 이미지 업로드 위젯
uploaded_file = st.file_uploader("이미지를 업로드하세요.", type=['jpg', 'png', 'jpeg'])


if uploaded_file is not None:
    # 업로드된 파일을 이미지로 변환
    image = Image.open(uploaded_file)
    review = scan_using_bedrock(image)
    print(review)
    