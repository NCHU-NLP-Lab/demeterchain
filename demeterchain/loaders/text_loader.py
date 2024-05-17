from typing import List, Iterator, Optional, Any
from pathlib import Path
from demeterchain.utils import Document


class TextLoader(object):
    """
    A class for loading text files from a directory using pathlib's glob function.

    Examples:

        .. code-block:: python

            from demeterchain.loaders import TextLoader

            # Load text files with default settings
            loader = TextLoader('/path/to/directory')
            documents = loader.load()

            # Load text files with custom glob pattern and show progress
            loader = TextLoader('/path/to/directory', '*.txt', show_progress=True)
            documents = loader.load()
    """
    def __init__(self, directory_path: str, glob: str = "[!.]*", show_progress: bool = False):
        """
        Initialize the TextLoader object with the specified directory path and settings.

        Args:
            directory_path : The path to the directory containing the text files.
            glob_pattern : The glob pattern to match the filenames. Defaults to '[!.]*'(all files except hidden).
            show_progress : Flag to determine whether to display progress information. Defaults to False.

        Returns:
            list: A list containing the contents of the text files.

        Example:
            >>> texts = textloader('/path/to/directory', '*.txt', show_progress=True)
        """
        self.directory_path = directory_path
        self.glob = glob
        self.show_progress = show_progress
    
    def load(self) -> List[Document]:
        """Load documents."""
        return list(self.lazy_load())

    def lazy_load(self) -> Iterator[Document]:
        """Load documents lazily."""
        p = Path(self.directory_path)
        if not p.exists():
            raise FileNotFoundError(f"Directory not found: '{self.directory_path}'")
        if not p.is_dir():
            raise ValueError(f"Expected directory, got file: '{self.directory_path}'")

        paths = p.glob(self.glob)
        items = [
            path 
            for path in paths
            if path.is_file()
        ]

        pbar = None
        if self.show_progress:
            try:
                from tqdm import tqdm

                pbar = tqdm(total=len(items))
            except ImportError as e:
                raise ImportError(
                    "To show the progress of TextLoader "
                    "you need to install tqdm, "
                    "`pip install tqdm`"
                )
        
        for i in items:
            yield from self._lazy_load_file(i, pbar)

        if pbar:
            pbar.close()

    def _lazy_load_file(
        self, 
        item: Path, 
        pbar: Optional[Any]
    ) -> Document:
        if item.is_file():
            try:
                page_content = item.read_text()
                metadata = {'source': str(item)}
                yield Document(page_content=page_content, metadata=metadata)
            except Exception as e:
                logger.error(f"Error loading file {str(item)}")
                raise e
            finally:
                if pbar:
                    pbar.update(1)