import boto3
import json
import uuid

# Initialize the Bedrock client
bedrock = boto3.client(service_name='bedrock')

# Define the request payload
payload = {
    "clientRequestToken": str(uuid.uuid4()),
    "description": "Inference profile for LLaMA model",
    "inferenceProfileName": "llama-inference-profile",
    "modelSource": {
       # "copyFrom": "arn:aws:bedrock:us-east-1:793002751149:inference-profile/us.meta.llama3-3-70b-instruct-v1:0"
        "copyFrom" : "arn:aws:bedrock:us-east-1::foundation-model/meta.llama3-3-70b-instruct-v1:0"
    },
    "tags": [
        {
            "key": "project",
            "value": "genai"
        }
    ]
}

# Create the inference profile
""" response = bedrock.create_inference_profile(
    clientRequestToken=payload["clientRequestToken"],
    description=payload["description"],
    inferenceProfileName=payload["inferenceProfileName"],
    modelSource=payload["modelSource"],
    tags=payload["tags"]
)
 """
# Retrieve the ARN of the created inference profile
""" inference_profile_arn = response["inferenceProfileArn"]
print(f"Inference Profile ARN: {inference_profile_arn}") """


prompt_data = """what is the capital of India?"""

bedrock_runtime = boto3.client(service_name='bedrock-runtime')

# Embed the prompt in Llama 3's instruction format.
formatted_prompt = f"""
<|begin_of_text|><|start_header_id|>user<|end_header_id|>
{prompt_data}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

runtime_payload = {
    "prompt": formatted_prompt,
    "max_gen_len": 90,
    "temperature": 0.5,
    "top_p": 0.9,
    #"inferenceProfileArn":inference_profile_arn
}

body = json.dumps(runtime_payload)
model_id = "meta.llama3-70b-instruct-v1:0"
#inference_profile_arn = "arn:aws:bedrock:us-east-1:793002751149:default-prompt-router/meta.llama:1"
runtime_response = bedrock_runtime.invoke_model(
    modelId=model_id, 
    body=body,
    contentType="application/json",
    accept="application/json",
    trace='ENABLED'
)

""" response_body = json.loads(runtime_response.get('body').decode('utf-8'))
print(response_body) """

# Read and parse the response body
response_body = json.loads(runtime_response['body'].read().decode('utf-8'))
print(json.dumps(response_body, indent=4))

# Extract the generated text from the response body
generated_text = response_body["generation"]
print(f"Generated Text: {generated_text
} ")    # Generated Text: The capital of India is New Delhi.