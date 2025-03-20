import boto3
import json
import uuid




prompt_data = """Act as a Shakespeare and write a poem on machine learning"""

bedrock = boto3.client(service_name='bedrock-runtime')

payload = {
   "messages": [
      {
        "role": "user",
        "content": [
          
          {
            "type": "text",
            "text": prompt_data
          }
        ]
      }
    ],
    "max_tokens": 30,
    "anthropic_version":"bedrock-2023-05-31"
}

body = json.dumps(payload)
model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
#inference_profile_arn = "arn:aws:bedrock:us-east-1:793002751149:default-prompt-router/meta.llama:1"
response = bedrock.invoke_model(
    modelId=model_id, 
    body=body,
    contentType="application/json",
    accept="application/json",
    trace='ENABLED'
)

#response_body = json.loads(response.get('body'))
print(response)