import abc
import logging
import os
from tqdm import tqdm
from typing import Dict, List
from docx import Document as DOCXDocument

from radlab_data.text.processors.splitter import SentenceSplitter


class PairElemsAbstract(abc.ABC):
    FIRST_PAIR_ELEM = None
    SECOND_PAIR_ELEM = None
    NAME = None
    ELEMS_TO_FILE_POSTFIX = None


class FirstSecondPairElems(PairElemsAbstract):
    FIRST_PAIR_ELEM = "first"
    SECOND_PAIR_ELEM = "second"
    NAME = f"{FIRST_PAIR_ELEM}_{SECOND_PAIR_ELEM}"
    ELEMS_TO_FILE_POSTFIX = {
        FIRST_PAIR_ELEM: "_FIRST.docx",
        SECOND_PAIR_ELEM: "_SECOND.docx",
    }


class FirstSecondNumberPairElems(PairElemsAbstract):
    FIRST_PAIR_ELEM = "1"
    SECOND_PAIR_ELEM = "2"
    NAME = f"{FIRST_PAIR_ELEM}_{SECOND_PAIR_ELEM}"
    ELEMS_TO_FILE_POSTFIX = {
        FIRST_PAIR_ELEM: "_1.docx",
        SECOND_PAIR_ELEM: "_2.docx",
    }


AVAILABLE_POSTFIXES = {
    FirstSecondPairElems.NAME: FirstSecondPairElems,
    FirstSecondNumberPairElems.NAME: FirstSecondPairElems,
}


class DirectoryPairsReader:
    AVAILABLE_POSTFIXES_STR = ", ".join(list(AVAILABLE_POSTFIXES.keys()))

    def __init__(
        self,
        dir_path: str | None = None,
        postfix_configuration_name: str = FirstSecondNumberPairElems.NAME,
    ):
        """
        Set directory path to read documents
        :param dir_path: Path to directory with documents to pair them
        """
        if postfix_configuration_name not in AVAILABLE_POSTFIXES:
            raise ValueError(
                f"Invalid postfix configuration: {postfix_configuration_name}. "
                f"Postfix configuration must be one of: "
                f"{self.AVAILABLE_POSTFIXES_STR}"
            )
        self.postfix: PairElemsAbstract = AVAILABLE_POSTFIXES[
            postfix_configuration_name
        ]()
        self._postfix_configuration = postfix_configuration_name

        self._documents_dir_path: str = dir_path

        # Base filename to documents pair (paths to both documents)
        self._document_pairs: Dict[str, Dict[str, str]] = {}
        self._document_pairs_content: Dict[str, Dict[List[str], List[str]]] = {}

        # base file name to dict: first/second to list of sentences
        self._document_pairs_sentences: Dict[str, Dict[List[str], List[str]]] = {}

        self._sentence_splitter = SentenceSplitter()

    @property
    def document_pairs_sentences(self):
        return self._document_pairs_sentences

    @property
    def base_name_to_paths(self):
        return self._document_pairs

    def prepare_pairs(
        self, dir_path: str | None = None
    ) -> Dict[str, Dict[str, str]]:
        """
        List files from given dir_path and prepare pairs of documents

        :param dir_path: Path to directory containing files (documents)
        :return: Dictionary with paths of pairs of documents
        """
        if dir_path is None:
            dir_path = self._documents_dir_path
        else:
            self._documents_dir_path = dir_path
        assert (
            dir_path is not None
        ), f"Directory with documents pairs is not provided"

        self._document_pairs.clear()
        self._document_pairs_content.clear()
        for file_name in os.listdir(dir_path):
            if not self._is_valid_filename(filename=file_name):
                logging.info(
                    f"Skipping {file_name} - is not valid "
                    f"{self._postfix_configuration} file name"
                )

            base_file_name = self._filename_without_postfix(file_name)
            if base_file_name not in self._document_pairs:
                self._document_pairs[base_file_name] = {
                    self.postfix.FIRST_PAIR_ELEM: "",
                    self.postfix.SECOND_PAIR_ELEM: "",
                }

            full_file_path = os.path.join(dir_path, file_name)
            if self._is_first_filename(file_name):
                self._document_pairs[base_file_name][
                    self.postfix.FIRST_PAIR_ELEM
                ] = full_file_path
            else:
                self._document_pairs[base_file_name][
                    self.postfix.SECOND_PAIR_ELEM
                ] = full_file_path
        return self._document_pairs

    def load_documents(self, unique_rows_in_cell: bool = False) -> None:
        """
        Loads documents from self._documents_pairs as read documents
        :param unique_rows_in_cell: If set to True, items from
        single row will be deduplicated
        :return:
        """
        assert len(self._document_pairs) > 0, "No documents provided to load!"

        self._document_pairs_content.clear()

        with tqdm(total=len(self._document_pairs), desc="Loading files") as pbar:
            for base_doc_name, doc_pair in self._document_pairs.items():
                first_paragraphs, second_paragraphs = None, None

                f_file = doc_pair[self.postfix.FIRST_PAIR_ELEM]
                s_file = doc_pair[self.postfix.SECOND_PAIR_ELEM]

                if f_file.endswith(".docx"):
                    first_paragraphs = self._load_docx_file_content(
                        file_path=f_file, unique_rows_in_cell=unique_rows_in_cell
                    )
                    second_paragraphs = self._load_docx_file_content(
                        file_path=s_file, unique_rows_in_cell=unique_rows_in_cell
                    )
                else:
                    logging.info(
                        f"Skipping file with not supported extension {f_file}"
                    )

                self._document_pairs_content[base_doc_name] = {
                    self.postfix.FIRST_PAIR_ELEM: first_paragraphs,
                    self.postfix.SECOND_PAIR_ELEM: second_paragraphs,
                }
                pbar.update()
        return self._split_documents_content_to_sentences()

    def _split_documents_content_to_sentences(self) -> None:
        assert (
            len(self._document_pairs_content) > 0
        ), "No documents provided to split content to sentences!"

        self._document_pairs_sentences.clear()
        with tqdm(
            total=len(self._document_pairs_content),
            desc="Splitting contents to sentences",
        ) as pbar:
            for base_doc_name, doc_pair in self._document_pairs_content.items():
                first_sentences = self._split_paragraphs_to_sentences(
                    doc_pair[self.postfix.FIRST_PAIR_ELEM]
                )
                second_sentences = self._split_paragraphs_to_sentences(
                    doc_pair[self.postfix.SECOND_PAIR_ELEM]
                )
                if not len(second_sentences) or not len(first_sentences):
                    logging.info(
                        f"Skipping empty document {base_doc_name} "
                        f"[len(first_sentences) = {len(first_sentences)}, "
                        f"len(second_sentences) = {len(second_sentences)}]"
                    )
                    continue

                self._document_pairs_sentences[base_doc_name] = {
                    self.postfix.FIRST_PAIR_ELEM: first_sentences,
                    self.postfix.SECOND_PAIR_ELEM: second_sentences,
                }
                pbar.update()
        return None

    def _split_paragraphs_to_sentences(
        self, paragraphs: list, min_sent_len: int = 1
    ):
        all_sentences = []
        for paragraph_text in paragraphs:
            sentences = self._sentence_splitter.split(
                paragraph_text, return_stats_dict=False
            )
            all_sentences.extend(sentences)
        return all_sentences

    @staticmethod
    def _load_docx_file_content(
        file_path: str, unique_rows_in_cell: bool = False
    ) -> List[str]:
        """
        Loads content from paragraphs and tables.
        :param file_path: path to file to load content
        :param unique_rows_in_cell: If set to True, items from
        single row will be deduplicated
        :return: List of read texts
        """
        docx_loader = DOCXDocument(file_path)
        par_texts = [p.text for p in docx_loader.paragraphs if len(p.text.strip())]
        tab_texts = []
        for table in docx_loader.tables:
            for row in table.rows:
                for cell in row.cells:
                    tab_texts.append(cell.text.strip())
        return par_texts + tab_texts

    def _filename_without_postfix(self, filename: str) -> str:
        for postfix in self.postfix.ELEMS_TO_FILE_POSTFIX.values():
            filename = filename.replace(postfix, "")
        return filename

    def _is_valid_filename(self, filename: str) -> bool:
        for postfix in self.postfix.ELEMS_TO_FILE_POSTFIX.values():
            if postfix in filename:
                return True
        return False

    def _is_first_filename(self, filename: str) -> bool:
        return filename.strip().endswith(
            self.postfix.ELEMS_TO_FILE_POSTFIX[self.postfix.FIRST_PAIR_ELEM]
        )
