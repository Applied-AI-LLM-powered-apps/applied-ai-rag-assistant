import base64
import hashlib
import imghdr
import json
import os
import io
import shutil
from typing import Optional, Union

import boto3
from langchain_core.documents import Document
from pypdf import PdfReader
from streamlit.runtime.uploaded_file_manager import UploadedFile

from utils.constants import ChunkType, Metadata, CollectionType
from utils.config_loader import load_config
from utils.utilsdoc import clean_text
from utils.utilsfile import put_file

config = load_config()

aws_profile_name = os.getenv("profile_name")
bedrock_region_name = config["BEDROCK"]["AWS_REGION_NAME"]
#bedrock_embeddings_model = config["BEDROCK"]["EMBEDDINGS_MODEL"]
bedrock_endpoint_url = config["BEDROCK"]["BEDROCK_ENDPOINT_URL"]
vision_model = config["VISION"]["VISION_MODEL"]

boto3.setup_default_session(profile_name=os.getenv("profile_name"))
bedrock = boto3.client("bedrock-runtime", bedrock_region_name, endpoint_url=bedrock_endpoint_url)



extract_image_output_dir = config['VISION']['IMAGE_OUTPUT_DIR']

def image_to_text(encoded_image, media_type) -> Optional[str]:
    system_prompt = """Describe every detail you can about this image,
        be extremely thorough and detail even the most minute aspects of the image.
        Start your description by providing an image title followed by a short overall summary.
        If the image is a table, output the content of the table in a structured format.
        """

    prompt = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1000,
        "temperature": 0,
        "system": system_prompt,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "data": encoded_image,
                            "media_type": media_type
                        }
                    },
                    {
                        "type": "text",
                        "text": system_prompt
                    }
                ]
            }
        ]
    }

    json_prompt = json.dumps(prompt)
    try:
        response = bedrock.invoke_model(body=json_prompt, modelId=vision_model,
                                        accept="application/json", contentType="application/json")
        response_body = json.loads(response.get('body').read())
        output = response_body['content'][0]['text']
        return output

    # Catch all other (unexpected) exceptions
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


def generate_unique_id(fname):
    # Generate MD5 hash of the filename
    hash_object = hashlib.md5(fname.name.encode())
    hex_dig = hash_object.hexdigest()
    return hex_dig


def load_image(pdfs: Union[list[UploadedFile], None, UploadedFile], metadata = None, restart_image_analysis:bool = False, ) -> Optional[list[Document]]:
    if pdfs is not None:
        docs = []
        if metadata is None:
            metadata = {}
        metadata.update({Metadata.CHUNK_TYPE.value: ChunkType.IMAGE.value})
        for pdf in pdfs:
            if pdf.type == "application/pdf":
                # Generate a unique identifier for each document
                tmp_id_based_on_file_upload = generate_unique_id(pdf)
                # Construct a save directory and create it
                save_dir = f"{extract_image_output_dir}/{tmp_id_based_on_file_upload}"
                if restart_image_analysis:
                    # Before processing is done, remove the directory and its contents
                    shutil.rmtree(save_dir)

                reader = PdfReader(pdf)

                os.makedirs(save_dir, exist_ok=True)

                for i, page in enumerate(reader.pages, start=1):
                    for image in page.images:

                        save_path = f"{save_dir}/{image.name}"
                        json_path = f"{save_dir}/{image.name}.json"

                        if os.path.exists(json_path):
                            # skip the image if it is already processed
                            with open(json_path, "r") as file:  # Open the document file
                                doc_data = json.load(file)  # Load the data from the document
                                # Create a new Document instance using the loaded data
                                doc = Document(page_content=doc_data['page_content'],  metadata=doc_data['metadata'])
                                docs.append(doc)  # Add the document to the docs list
                            continue

                        with open(save_path, "wb") as fp:
                            fp.write(image.data)

                        # Determine image type
                        image_type = imghdr.what(save_path)
                        media_type = f"image/{image_type}"

                        image_content = encode_image(save_path)
                        image_description = image_to_text(image_content, media_type)
                        if image_description is not None:
                            page_metadata = {'page': i, 'filename': pdf.name, 'media_type': media_type}
                            page_metadata.update(metadata)
                            doc = Document(page_content=clean_text(image_description), metadata=page_metadata)
                            docs.append(doc)

                            with open(json_path, "w") as file:
                                json.dump(doc.__dict__, file)

                        else:
                            print(f"Failed to extract text from image {image.name}.")

                        put_file(io.BytesIO(image.data), image.name, CollectionType.IMAGES.value)
        return docs
    else:
        return None


def encode_image(image_path):
    """Function to encode images"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
