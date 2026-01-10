import os
from multiprocessing import Pool

import tqdm
import pathlib
import logging
import queue

from radlab_data.text.document import Document
from radlab_data.text.loaders.config import AVAILABLE_FILE_EXTENSIONS


class DirectoryFileReader:
    def __init__(
        self,
        main_dir_path: str,
        read_sub_dirs: bool = True,
        accept_extensions: list = AVAILABLE_FILE_EXTENSIONS,
        processes_count: int = None,
    ):
        """
        :param main_dir_path: Path to main directory to load files
        :param read_sub_dirs: If set to True, then will be done listing
        all files in main_dir_path with subdirectories
        :param accept_extensions: List of accepted extension
        :param processes_count: If number of processes is greater than 1,
        will use given number of processes to load listed files.
        """
        self.main_dir_path = main_dir_path
        self.deep_read = read_sub_dirs
        self.accept_extensions = accept_extensions

        self._processes_count = processes_count
        self._mt_queue = None
        if processes_count is not None and processes_count > 1:
            self._mt_queue = queue.Queue()
        self._files_in_directory = self._list_files()
        self._all_documents_in_directory = []

    @property
    def documents(self) -> list:
        return self._all_documents_in_directory

    def load(
        self,
        prepare_proper_pages: bool = False,
        merge_document_pages: bool = False,
        clear_texts: bool = False,
        use_text_denoiser: bool = False,
        max_tokens_in_chunk: int = None,
        number_of_overlap_tokens: int = None,
        check_text_language: bool = False,
    ) -> None:
        """
        Loads all documents listed in `self._files_in_directory`
        :param prepare_proper_pages: IF set to True, then all texts/sentences
        from the same page will be merged to single page.
        :param merge_document_pages: If set to True then all pages from document
        will be merged to single page, then document will contain single page.
        :param clear_texts: If set to True, then each text will be clear
        :param use_text_denoiser: If set to True, then each text will be denoised
        :param max_tokens_in_chunk: If given then each text will be split
        to chunks with given max number of tokens in single chunk
        :param number_of_overlap_tokens: If given then each  generated chunk
        will contain number_of_overlap_tokens from the previous chunk.
        :param check_text_language: If options is set to True, then language
        of each text will be checked and stored to metadata
        :return: None
        """
        logging.info(f"processes_count={self._processes_count}")
        logging.info(f"prepare_proper_pages={prepare_proper_pages}")
        logging.info(f"merge_document_pages={merge_document_pages}")
        logging.info(f"clear_texts={clear_texts}")
        logging.info(f"use_text_denoiser={use_text_denoiser}")
        logging.info(f"check_text_language={check_text_language}")
        logging.info(f"max_tokens_in_chunk={max_tokens_in_chunk}")
        logging.info(f"number_of_overlap_tokens={number_of_overlap_tokens}")

        if self._mt_queue is not None:
            return self._load_mt(
                prepare_proper_pages=prepare_proper_pages,
                merge_document_pages=merge_document_pages,
                clear_texts=clear_texts,
                use_text_denoiser=use_text_denoiser,
                max_tokens_in_chunk=max_tokens_in_chunk,
                number_of_overlap_tokens=number_of_overlap_tokens,
                check_text_language=check_text_language,
            )

        with tqdm.tqdm(
            total=len(self._files_in_directory), desc="Loading files"
        ) as pbar:
            for file_path in self._files_in_directory:
                try:
                    f_spl = file_path.split(self.main_dir_path)
                    relative_file_path = f_spl[1] if len(f_spl) > 1 else f_spl[0]

                    doc = Document(
                        file_path=file_path,
                        relative_file_path=relative_file_path,
                        prepare_proper_pages=prepare_proper_pages,
                        merge_document_pages=merge_document_pages,
                        clear_texts=clear_texts,
                        use_text_denoiser=use_text_denoiser,
                        max_tokens_in_chunk=max_tokens_in_chunk,
                        number_of_overlap_tokens=number_of_overlap_tokens,
                        check_text_language=check_text_language,
                    )
                    doc.load()
                    self._all_documents_in_directory.append(doc)
                except Exception as e:
                    logging.warning(f"Problem wih loading file {file_path}")
                    logging.warning(e)
                pbar.update()

    def _list_files(self) -> list:
        """
        Depends on the self.deep_read parameter, reads all files in main_dir_path
        and subdirectories when set to True, otherwise reads only from main directory.
        :return: List of paths to files to read
        """
        if self.deep_read:
            all_files_to_read = pathlib.Path(self.main_dir_path)
            all_files_to_read = [str(d) for d in all_files_to_read.rglob("*")]
        else:
            all_files_to_read = [
                os.path.join(self.main_dir_path, file_name)
                for file_name in os.listdir(self.main_dir_path)
            ]
        filtered_files = []
        for file_path in all_files_to_read:
            if os.path.isdir(file_path):
                continue
            ext = file_path.split(".")[-1].lower().strip()
            if ext not in self.accept_extensions:
                logging.warning("Skipped file {}".format(file_path))
                continue
            filtered_files.append(file_path)
        return filtered_files

    @staticmethod
    def _load_document_mt(document: Document):
        """
        Document to load into thread, object of Document may be returned
        by this method because may be pickled!
        :param document: Document to load
        :return: Loaded document
        """
        document.load()
        return document

    def _load_mt(
        self,
        prepare_proper_pages: bool = False,
        merge_document_pages: bool = False,
        clear_texts: bool = False,
        use_text_denoiser: bool = False,
        max_tokens_in_chunk: int = None,
        number_of_overlap_tokens: int = None,
        check_text_language: bool = False,
    ) -> None:
        pool = Pool(self._processes_count)
        async_loading = []
        for file_path in self._files_in_directory:
            f_spl = file_path.split(self.main_dir_path)
            relative_file_path = f_spl[1] if len(f_spl) > 1 else f_spl[0]

            doc = Document(
                file_path=file_path,
                relative_file_path=relative_file_path,
                prepare_proper_pages=prepare_proper_pages,
                merge_document_pages=merge_document_pages,
                clear_texts=clear_texts,
                use_text_denoiser=use_text_denoiser,
                max_tokens_in_chunk=max_tokens_in_chunk,
                number_of_overlap_tokens=number_of_overlap_tokens,
                check_text_language=check_text_language,
            )
            async_loading.append(
                pool.apply_async(self._load_document_mt, args=(doc,))
            )
        with tqdm.tqdm(total=len(async_loading), desc="Loading files") as pbar:
            for worker in async_loading:
                try:
                    wd = worker.get()
                    self._all_documents_in_directory.append(wd)
                except Exception as e:
                    logging.warning("Problem wih loading file")
                    logging.warning(e)
                pbar.update()
        pool.close()
        pool.join()
