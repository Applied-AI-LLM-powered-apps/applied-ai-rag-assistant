from enum import Enum

class DocumentType(Enum):
    STANDARD = "Norme"
    GUIDE = "Guide"
    TUTORIAL = "Tutoriel"
    FAQ = "FAQ"


class ChunkType(Enum):
    TEXT = "Text"
    IMAGE = "Image"


class Metadata(Enum):
    DOCUMENT_TYPE = "document_type"
    CHUNK_TYPE = "chunk_type"
    TOPIC = "topic"
    PAGE = "page"
    FILENAME = "filename"


class SupportedFileType(Enum):
    PDF = "pdf"
    MARKDOWN = "md"
    TEXT = "txt"

class StorageType(Enum):
    S3 = "S3"
    LOCAL = "LOCAL"
    NONE = "NONE"


class CollectionType(Enum):
    DOCUMENTS = "documents"
    IMAGES = "images"
