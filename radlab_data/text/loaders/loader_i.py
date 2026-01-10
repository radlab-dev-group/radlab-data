import sys
import abc
import copy

from transformers import AutoTokenizer

from radlab_data.text.utils import TextUtils
from radlab_data.text.loaders.config import TOKENIZER_PATH
from radlab_data.text.processors.cleaner import FullChainProcessor


class LoaderI(abc.ABC):
    def __init__(self, filepath: str, options: dict = None) -> None:
        """
        :param filepath: Path to docx file
        """
        self.filepath = filepath
        self.options = {} if not options else options
        self.doc_pages = []
        self.text_chunk_type = None
        self._text_cleaner = FullChainProcessor(phrases_to_remove=[])
        self._tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
        self._tokenizer.model_max_length = sys.maxsize

        self._check_language = options.get("check_text_language", False)
        if self._check_language is None:
            self._check_language = False

        self._resolve_text_chunk_type()

    def _resolve_text_chunk_type(self):
        sep = ""
        self.text_chunk_type = "raw_text"
        if "clear_texts" in self.options:
            clr = self.options.get("clear_texts", False)
            if clr is not None and clr:
                self.text_chunk_type = "clear_texts"
        if "use_text_denoiser" in self.options:
            utd = self.options.get("use_text_denoiser", False)
            if utd is not None and utd:
                sep = "_"
                self.text_chunk_type += f"{sep}denoised"
        if "merge_to_proper_pages" in self.options:
            mp = self.options.get("merge_to_proper_pages", False)
            if mp is not None and mp:
                sep = "_"
                self.text_chunk_type += f"{sep}proper_page"
        if "merge_document_pages" in self.options:
            md = self.options.get("merge_document_pages", False)
            if md is not None and md:
                sep = "_"
                self.text_chunk_type += f"{sep}merged_document"
        if "max_tokens_in_chunk" in self.options:
            mt = self.options.get("max_tokens_in_chunk", 0)
            if mt is not None and mt > 1:
                sep = "_"
                self.text_chunk_type += f"{sep}chunk_max_tokens_{mt}"

    def as_dict(self) -> dict:
        """
        Return document as dictionary (json)
        :return: Document converted to dictionary
        """
        relative_file_path = self.filepath
        if "relative_file_path" in self.options:
            relative_file_path = self.options.pop("relative_file_path")

        category = ""
        if "document_category" in self.options:
            category = self.options.pop("document_category")

        doc_dict = {
            "filepath": self.filepath,
            "relative_filepath": relative_file_path,
            "category": category,
            "options": self.options,
            "pages": [],
        }
        for doc_page in self.doc_pages:
            if "text_chunk_type" not in doc_page.metadata:
                doc_page.metadata["text_chunk_type"] = self.text_chunk_type
            doc_dict["pages"].append(
                {
                    "page_number": doc_page.metadata.get("page", 0),
                    "table_number": doc_page.metadata.get("table", 0),
                    "row_number": doc_page.metadata.get("row", 0),
                    "column_number": doc_page.metadata.get("column", 0),
                    "page_content": doc_page.page_content,
                    "metadata": doc_page.metadata,
                }
            )
        return doc_dict

    def merge_all_pages(self) -> None:
        """
        Of option to merging is set as true, all pages are
        concatenated as single. Actual self.doc_pages is replaced
        with list with the single page (whole concatenated document)
        :return: None
        """
        if not self.options.get("merge_document_pages", False) or not len(
            self.doc_pages
        ):
            return None

        merged_pages = ""
        for doc_page in self.doc_pages:
            merged_pages += doc_page.page_content + "\n"
        document_page = self.doc_pages[0]
        document_page.metadata["page"] = 1
        document_page.page_content = merged_pages
        self.doc_pages = [document_page]

    def clear_doc_pages(self) -> None:
        """
        If clear_texts options is given, then all pages will be cleared
        using FullChainProcess cleaner. Cleaning is done in place.
        :return: None
        """
        if not self.options.get("clear_texts", False):
            return None

        for doc_page in self.doc_pages:
            doc_page.page_content = self._text_cleaner.apply(doc_page.page_content)

    def split_to_chunks(self) -> None:
        """
        Each page will be split into chunks of given size. Additionally,
        generated chunks will be overlapped if option is set.
        :return: None
        """
        chunk_size = self.options.get("max_tokens_in_chunk", 0)
        overlap_size = self.options.get("number_of_overlap_tokens", 0)
        if overlap_size is not None and overlap_size > 0:
            self.text_chunk_type += f"_overlap_{overlap_size}"

        if chunk_size is None or chunk_size < 2:
            return None

        new_pages = []
        for doc_page in self.doc_pages:
            new_pages.extend(
                self._split_page_to_chunks_max_tokens(
                    doc_page=doc_page,
                    chunk_size=chunk_size,
                    overlap_size=overlap_size,
                )
            )
        self.doc_pages = new_pages

    def check_language(self) -> None:
        if not self._check_language:
            return None

        for doc_page in self.doc_pages:
            doc_page.metadata["language"] = TextUtils.text_language(
                text_str=doc_page.page_content, probs=False
            )

    @staticmethod
    def chunks(tokenized_page, chunk_size, overlap_size):
        step_size = chunk_size
        if overlap_size is not None and overlap_size > 1:
            step_size = chunk_size - overlap_size
        for index in range(0, len(tokenized_page), step_size):
            yield tokenized_page[index : index + chunk_size]

    def _split_page_to_chunks_max_tokens(
        self, doc_page, chunk_size, overlap_size
    ) -> list:
        """
        Split page content to chunks of given max_tokens_in_chunk size
        :param doc_page: Document to split
        :param chunk_size: Chunk size (max number of tokens)
        :param overlap_size: Number of tokens overlapping adjacent chunks
        :return: List of objects of Document (deepcopy of doc_page
        with tokenized and chunked document page content)
        """
        if chunk_size < 1:
            return [doc_page]

        # NOTE:
        # This is not needed because tokenizer max model len is set as sys.maxsize
        # If this option will not be available, teb uncomment:
        # chunked_page_content_str = self._split_page_content_to_max_model_len(
        #     doc_page.page_content
        # )
        # ...and comment the line below
        chunked_page_content_str = [doc_page.page_content]

        all_chunks = []
        chunk_number = 1
        for chunk_str in chunked_page_content_str:
            tokenized_page = self._tokenizer(chunk_str).input_ids
            for chunk in self.chunks(tokenized_page, chunk_size, overlap_size):
                chunk_doc = copy.deepcopy(doc_page)
                chunk_doc.page_content = self._tokenizer.decode(chunk)
                chunk_doc.metadata["chunk"] = chunk_number
                all_chunks.append(chunk_doc)
                chunk_number += 1
        return all_chunks

    def _split_page_content_to_max_model_len(self, page_content_str: str) -> list:
        """
        When the page content is longer than the maximum model sequence length,
        then the text will be split into chunks of that maximum model length.

        In case when chunk of text ends on the non-whitespace character then
        this chunk will be cut to the first whitespace character before
        the end-position (max model length) of processed page text.


        :param page_content_str: Page content as string
        :return: List of chunks of page content with maximum model length
        """
        max_text_len = self._tokenizer.model_max_length
        page_content_len = len(page_content_str)
        if page_content_len <= max_text_len:
            return [page_content_str]

        content_chunks = []

        begin_position = 0
        end_position = max_text_len

        # last 5% of data will be used as slicing context
        slice_left_size = int(max_text_len * 0.05)

        while True:
            if (begin_position + slice_left_size) >= page_content_len:
                break
            page_chunk = page_content_str[begin_position:end_position]
            content_chunks.append(page_chunk)
            begin_position = end_position - slice_left_size
            end_position = begin_position + max_text_len
            if begin_position + max_text_len >= page_content_len:
                end_position = page_content_len
        return content_chunks
