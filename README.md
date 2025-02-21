# AWS Bedrock
## LLM Model
### LLAMA 3

This project demonstrates how to use AWS Bedrock to interact with various large language models (LLMs) such as LLAMA 3 and Claude 2. The project includes examples of invoking these models using the AWS SDK for Python (Boto3).

### Prerequisites

- Python 3.8 or higher
- AWS SDK for Python (Boto3)
- AWS credentials configured

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/aws-bedrock.git
    cd aws-bedrock
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    .\venv\Scripts\activate  # On Windows
    source venv/bin/activate  # On macOS/Linux
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

### Usage

#### Invoking LLAMA 3 Model

1. Open `llama2.py` and update the `model_id` and `inferenceProfileArn` with the appropriate values.
2. Run the script:
    ```sh
    python llama2.py
    ```
#### Invoking Claude 2 Model

1. Open [claude2.py](http://_vscodecontentref_/1) and update the `model_id` and `inferenceProfileArn` with the appropriate values.
2. Run the script:
    ```sh
    python claude2.py
    ```    
### License

This project is licensed under the MIT License. See the LICENSE file for details.

### Acknowledgements

- [AWS Bedrock](https://aws.amazon.com/bedrock/)
- [Boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)
- [LLAMA - 3 user guide](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-meta.html)
- [AWS Bedrock InvokeModel](https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-runtime_example_bedrock-runtime_InvokeModel_MetaLlama3_section.html)
- [AWS LLAMA 3-3-70b Model](https://us-east-1.console.aws.amazon.com/bedrock/home?region=us-east-1#/model-catalog/serverless/meta.llama3-3-70b-instruct-v1:0)
- [AWS Bedrock Cloude model anthropic.claude-v2:1](https://us-east-1.console.aws.amazon.com/bedrock/home?region=us-east-1#/model-catalog/serverless/anthropic.claude-v2:1)
- [Cloude v2 model user guide](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-text-completion.html)



