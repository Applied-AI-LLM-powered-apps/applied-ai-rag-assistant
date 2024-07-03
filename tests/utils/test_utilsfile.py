"""Test the utilsfile file."""

import unittest
import os
#from pytest import fixture
from unittest.mock import patch
from rag_assistant.utils.utilsfile import list_files, _list_files_locally, _list_files_from_s3


class TestListFiles(unittest.TestCase):
    """Test the list_files function."""

    @patch('rag_assistant.utils.utilsfile.config.get')
    @patch('rag_assistant.utils.utilsfile._list_files_locally')
    def test_list_files_locally(self, mock_list_files_locally, mock_config_get):
        """Test list_files with LOCAL configuration."""
        mock_config_get.return_value = 'LOCAL'
        mock_list_files_locally.return_value = ['file1.txt', 'file2.txt']
        result = list_files('my_collection')
        self.assertEqual(result, ['file1.txt', 'file2.txt'])
        mock_list_files_locally.assert_called_once_with(file_collection='my_collection')

    @patch('rag_assistant.utils.utilsfile.config.get')
    @patch('rag_assistant.utils.utilsfile._list_files_from_s3')
    def test_list_files_s3(self, mock_list_files_from_s3, mock_config_get):
        """Test list_files with S3 configuration."""
        mock_config_get.return_value = 'S3'
        mock_list_files_from_s3.return_value = ['file1.txt', 'file2.txt']
        result = list_files('my_collection')
        self.assertEqual(result, ['file1.txt', 'file2.txt'])
        mock_list_files_from_s3.assert_called_once_with(file_collection='my_collection')

    @patch('rag_assistant.utils.utilsfile.config.get')
    def test_list_files_none(self, mock_config_get):
        """Test list_files with NONE configuration."""
        mock_config_get.return_value = 'NONE'
        result = list_files('my_collection')
        self.assertIsNone(result)

    @patch('rag_assistant.utils.utilsfile.config.get')
    def test_list_files_not_implemented(self, mock_config_get):
        """Test list_files with an unknown configuration."""
        mock_config_get.return_value = 'UNKNOWN'
        with self.assertRaises(NotImplementedError):
            list_files('my_collection')

    @patch('rag_assistant.utils.utilsfile.config.get')
    @patch('os.path.exists')
    @patch('os.listdir')
    def test__list_files_locally(self, mock_listdir, mock_path_exists, mock_config_get):
        """Test _list_files_locally function."""
        mock_listdir.return_value = ['file1.txt', 'file2.txt', 'file3.jpg']
        mock_path_exists.return_value = True
        mock_config_get.return_value = 'data'
        result = _list_files_locally('my_local_collection')
        self.assertEqual(result, ['file1.txt', 'file2.txt', 'file3.jpg'])
        mock_listdir.assert_called_once_with(os.path.join('data', 'my_local_collection'))
        mock_path_exists.assert_called_once_with(os.path.join('data', 'my_local_collection'))

    @patch('rag_assistant.utils.utilsfile.config.get')
    @patch('os.path.exists')
    @patch('os.listdir')
    def test__list_files_locally_empty(self, mock_listdir, mock_path_exists, mock_config_get):
        """Test _list_files_locally function with an empty directory."""
        mock_listdir.return_value = []
        mock_path_exists.return_value = True
        mock_config_get.return_value = 'data'
        result = _list_files_locally('empty_collection')
        self.assertEqual(result, [])
        mock_listdir.assert_called_once_with(os.path.join('data', 'empty_collection'))

    @patch('rag_assistant.utils.utilsfile.boto3.client')
    @patch('rag_assistant.utils.utilsfile.config.get')
    def test__list_files_from_s3(self, mock_config_get, mock_boto3_client):
        """Test _list_files_from_s3 function."""
        mock_config_get.return_value = 'my_test_bucket'
        mock_s3_client = mock_boto3_client.return_value
        mock_s3_client.list_objects_v2.return_value = {
            'Contents': [
                {'Key': 'file_collection/file1.txt'},
                {'Key': 'file_collection/file2.txt'}
            ]
        }
        result = _list_files_from_s3('file_collection')
        self.assertEqual(result, ['file1.txt', 'file2.txt'])
        mock_boto3_client.assert_called_once_with('s3')
        mock_s3_client.list_objects_v2.assert_called_once_with(Bucket='my_test_bucket',
                                                               Prefix='file_collection')

    @patch('rag_assistant.utils.utilsfile.boto3.client')
    @patch('rag_assistant.utils.utilsfile.config.get')
    def test__list_files_from_s3_empty(self, mock_config_get, mock_boto3_client):
        """Test _list_files_from_s3 function with an empty collection."""
        mock_config_get.return_value = 'my_test_bucket'
        mock_s3_client = mock_boto3_client.return_value
        mock_s3_client.list_objects_v2.return_value = {}
        result = _list_files_from_s3('empty_collection')
        self.assertEqual(result, [])
        mock_boto3_client.assert_called_once_with('s3')
        mock_s3_client.list_objects_v2.assert_called_once_with(Bucket='my_test_bucket',
                                                               Prefix='empty_collection')

    @patch('rag_assistant.utils.utilsfile.boto3.client')
    @patch('rag_assistant.utils.utilsfile.config.get')
    def test__list_files_from_s3_exception(self, mock_config_get, mock_boto3_client):
        """Test _list_files_from_s3 function with an exception."""
        mock_config_get.return_value = 'my_test_bucket'
        mock_s3_client = mock_boto3_client.return_value
        mock_s3_client.list_objects_v2.side_effect = Exception('S3 access error')
        with self.assertRaises(Exception) as context:
            _list_files_from_s3('invalid_collection')
        self.assertTrue('S3 access error' in str(context.exception))
