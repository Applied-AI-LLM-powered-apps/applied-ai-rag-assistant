"""Utils for storing files"""
import io
import os
import logging
import boto3

from streamlit.runtime.uploaded_file_manager import UploadedFile

from .config_loader import load_config
from .constants import StorageType

config = load_config()

logger = logging.getLogger(__name__)

def put_file(file: io.BytesIO, filename:str, file_collection: str = '') -> None:
    """Persist file to selected storage interface"""
    storage_interface = config.get('DOCUMENTS_STORAGE', 'INTERFACE')

    if storage_interface == StorageType.LOCAL.value:
        _persist_file_locally(file, filename=filename, file_collection=file_collection)
    elif storage_interface == StorageType.S3.value:
        _persist_file_to_s3(file, filename=filename, file_collection=file_collection)
    elif storage_interface == StorageType.NONE.value:
        pass
    else:
        raise NotImplementedError(f"{storage_interface} not implemented yet for storage.")


def get_file(filename: str, file_collection: str=''):
    """Get file from selected storage interface"""
    storage_interface = config.get('DOCUMENTS_STORAGE', 'INTERFACE')

    if storage_interface == StorageType.LOCAL.value:
        return _get_file_locally(filename=filename, file_collection=file_collection)

    if storage_interface == StorageType.S3.value:
        return _get_file_from_s3(filename=filename, file_collection=file_collection)

    if storage_interface == StorageType.NONE.value:
        return None

    raise NotImplementedError(f"{storage_interface} not implemented yet for storage.")


def delete_file(filename: str, file_collection: str=''):
    """Delete file from selected storage interface"""
    storage_interface = config.get('DOCUMENTS_STORAGE', 'INTERFACE')

    if storage_interface == StorageType.LOCAL.value:
        return _delete_file_locally(filename=filename, file_collection=file_collection)

    if storage_interface == StorageType.S3.value:
        return _delete_file_from_s3(filename=filename, file_collection=file_collection)

    if storage_interface == StorageType.NONE.value:
        return None

    raise NotImplementedError(f"{storage_interface} not implemented yet for storage.")


def list_files(file_collection: str=''):
    """List files from selected storage interface"""
    storage_interface = config.get('DOCUMENTS_STORAGE', 'INTERFACE')

    if storage_interface == StorageType.LOCAL.value:
        return _list_files_locally(file_collection=file_collection)

    if storage_interface == StorageType.S3.value:
        return _list_files_from_s3(file_collection=file_collection)

    if storage_interface == StorageType.NONE.value:
        return None

    raise NotImplementedError(f"{storage_interface} not implemented yet for storage.")


def _persist_file_to_s3(file: io.BytesIO, filename: str, file_collection: str) -> None:
    """Persist file to S3 storage"""
    logger.info("On persiste un document : %s sur S3", filename)
    s3_bucket = config.get('DOCUMENTS_STORAGE', 'S3_BUCKET_NAME')

    file_key = f"{file_collection}/{filename}"

    s3_client = boto3.client('s3')

    s3_client.upload_fileobj(file, s3_bucket, file_key)


def _persist_file_locally(file: io.BytesIO, filename:str, file_collection: str) -> None:
    """Persist file to local storage"""
    logger.info("On persiste un document : %s localement", filename)
    documents_path = config.get('DOCUMENTS_STORAGE', 'DOCUMENTS_PATH')

    file_path = os.path.join(documents_path, file_collection)

    if not os.path.exists(file_path):
        os.makedirs(file_path)

    file_path = os.path.join(file_path, filename)

    with open(file_path, 'wb') as f:
        f.write(file.getbuffer())


def _get_file_locally(filename: str, file_collection: str):
    """Get file from local storage"""
    documents_path = config.get('DOCUMENTS_STORAGE', 'DOCUMENTS_PATH')

    file_path = os.path.join(documents_path, file_collection, filename)

    if os.path.exists(file_path):
        return open(file_path, 'rb').read()

    return None


def _get_file_from_s3(filename: str, file_collection: str):
    """Get file from S3 storage"""
    s3_bucket = config.get('DOCUMENTS_STORAGE', 'S3_BUCKET_NAME')

    file_key = f"{file_collection}/{filename}"

    s3_client = boto3.client('s3')

    response = s3_client.get_object(Bucket=s3_bucket, Key=file_key)
    return response['Body'].read()


def _delete_file_locally(filename: str, file_collection: str):
    """Delete file from local storage"""
    documents_path = config.get('DOCUMENTS_STORAGE', 'DOCUMENTS_PATH')

    file_path = os.path.join(documents_path, file_collection, filename)

    if os.path.exists(file_path):
        os.remove(file_path)


def _delete_file_from_s3(filename: str, file_collection: str):
    """Delete file from S3 storage"""
    s3_bucket = config.get('DOCUMENTS_STORAGE', 'S3_BUCKET_NAME')

    file_key = f"{file_collection}/{filename}"

    s3_client = boto3.client('s3')

    s3_client.delete_object(Bucket=s3_bucket, Key=file_key)


def _list_files_locally(file_collection: str):
    """List files from local storage"""
    documents_path = config.get('DOCUMENTS_STORAGE', 'DOCUMENTS_PATH')

    file_path = os.path.join(documents_path, file_collection)

    if os.path.exists(file_path):
        return os.listdir(file_path)

    return []


def _list_files_from_s3(file_collection: str):
    """List files from S3 storage"""
    s3_bucket = config.get('DOCUMENTS_STORAGE', 'S3_BUCKET_NAME')

    s3_client = boto3.client('s3')

    response = s3_client.list_objects_v2(Bucket=s3_bucket, Prefix=file_collection)

    if 'Contents' in response:
        return [obj['Key'].split('/')[-1] for obj in response['Contents']]

    return []
