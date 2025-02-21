import boto3
import json
import uuid

# Initialize the Bedrock client

bedrock_runtime = boto3.client(service_name='bedrock-runtime')


prompt_data = "Act as a Shakespeare and write a poem on PHP"
formatted_prompt = f"\n\nHuman: {prompt_data}\n\nAssistant:"

runtime_payload = {
    "prompt": formatted_prompt,
    "max_tokens_to_sample": 90,
    "temperature": 0.5,
    "top_p": 0.9,
    #"inferenceProfileArn":inference_profile_arn
}

body = json.dumps(runtime_payload)
model_id = "anthropic.claude-v2:1"
runtime_response = bedrock_runtime.invoke_model(
    modelId=model_id, 
    body=body,
    contentType="application/json",
    accept="application/json",
    trace='ENABLED'
)

response_body = json.loads(runtime_response['body'].read().decode('utf-8'))
print(json.dumps(response_body, indent=4))

# Extract the generated text from the response body
generated_text = response_body["completion"]
print(f"Generated Text: {generated_text
} ")    # Generated Text: The capital of India is New Delhi.